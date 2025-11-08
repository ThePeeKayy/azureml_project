// app/api/train-cnn/route.ts
import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs";

interface HyperParameters {
  modelType: string;
  learningRate: number;
  batchSize: number;
  numSamples: number;
}

async function getAzureMLAuthToken(): Promise<string> {
  const tenantId = process.env.AZURE_TENANT_ID!;
  const tokenUrl = `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/token`;
  const params = new URLSearchParams({
    client_id: process.env.AZURE_CLIENT_ID!,
    client_secret: process.env.AZURE_CLIENT_SECRET!,
    grant_type: "client_credentials",
    scope: "https://management.azure.com/.default",
  });
  const response = await fetch(tokenUrl, { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: params });
  if (!response.ok) throw new Error("Azure auth failed");
  return (await response.json()).access_token;
}

async function submitAzureMLJob(params: HyperParameters) {
  const token = await getAzureMLAuthToken();
  const scriptPath = path.join(process.cwd(), "scripts", "train_cnn.py");
  const scriptContent = fs.readFileSync(scriptPath, "utf-8");
  const jobId = `cnn-training-${Date.now()}`;
  const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID!;
  const resourceGroup = process.env.AZURE_RESOURCE_GROUP!;
  const workspaceName = process.env.AZURE_ML_WORKSPACE_NAME!;
  const apiUrl = `https://management.azure.com/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/jobs/${jobId}?api-version=2024-01-01-preview`;
  const jobConfig = {
    properties: {
      jobType: "Command",
      command: `cat > train_cnn.py << 'EOFSCRIPT'
${scriptContent}
EOFSCRIPT
python train_cnn.py --model-type ${params.modelType} --lr ${params.learningRate} --batch-size ${params.batchSize} --epochs 2 --num-samples ${params.numSamples} --dataset cifar10`,
      environmentId: `/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/environments/AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu/versions/10`,
      computeId: `/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/computes/cpu-cluster`,
      experimentName: "cnn-training",
      displayName: `Fast CNN Training - ${params.modelType} (${params.numSamples} samples) - ${new Date().toISOString()}`,
      description: `Fast CNN training on CIFAR-10 dataset`,
      environmentVariables: {AZURE_STORAGE_CONNECTION_STRING: process.env.AZURE_STORAGE_CONNECTION_STRING || ""},
    },
  };
  const response = await fetch(apiUrl, { method: "PUT", headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }, body: JSON.stringify(jobConfig) });
  if (!response.ok) throw new Error("Job submission failed");
  return { jobId, status: "submitted" };
}

async function getBlobUrls(jobId: string, maxRetries: number = 90, delayMs: number = 10000) {
  const connectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
  if (!connectionString) return null;
  const accountNameMatch = connectionString.match(/AccountName=([^;]+)/);
  if (!accountNameMatch) return null;
  const accountName = accountNameMatch[1];
  const baseUrl = `https://${accountName}.blob.core.windows.net/model-outputs/${jobId}`;
  const manifestUrl = `${baseUrl}/manifest.json`;
  
  // Retry logic to wait for manifest.json to be created
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const manifestResponse = await fetch(manifestUrl);
      
      if (manifestResponse.ok) {
        const manifest = await manifestResponse.json();
        
        // Extract URLs from manifest
        return {
          model_url: manifest.model_url || null,
          results_url: manifest.results_url || null,
          metrics_url: manifest.metrics_url || null,
          manifest_url: manifestUrl,
          portal_url: baseUrl
        };
      }
    } catch (error) {
      console.log(`Attempt ${attempt + 1}: manifest.json not ready yet`);
    }
    
    // Wait before retrying
    await new Promise(resolve => setTimeout(resolve, delayMs));
  }
  
  // Return URLs even if manifest not found after all retries
  return {
    model_url: null,
    results_url: null,
    metrics_url: null,
    manifest_url: manifestUrl,
    portal_url: baseUrl
  };
}

async function* streamAzureMLJobProgress(jobId: string) {
  const token = await getAzureMLAuthToken();
  const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID!;
  const resourceGroup = process.env.AZURE_RESOURCE_GROUP!;
  const workspaceName = process.env.AZURE_ML_WORKSPACE_NAME!;
  const apiUrl = `https://management.azure.com/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/jobs/${jobId}?api-version=2024-01-01-preview`;
  let previousStatus = "";

  while (true) {
    try {
      const response = await fetch(apiUrl, { method: "GET", headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" } });
      if (!response.ok) throw new Error(`Failed to fetch job status`);

      const job = await response.json();
      const status = job.properties?.status || "unknown";

      if (status !== previousStatus) {
        yield JSON.stringify({ jobId, status: status.toLowerCase(), timestamp: new Date().toISOString() }) + "\n";
        previousStatus = status;
      }

      if (status === "Completed") {
        yield JSON.stringify({ jobId, status: "waiting_for_outputs", timestamp: new Date().toISOString() }) + "\n";
        const urls = await getBlobUrls(jobId); // keep maxRetries=20 or more
        yield JSON.stringify({ jobId, status: "completed", results: urls, timestamp: new Date().toISOString() }) + "\n";
        break;
      }



      if (status === "Failed" || status === "Canceled") {
        yield JSON.stringify({ jobId, status: "error", error: `Job ${status.toLowerCase()}`, timestamp: new Date().toISOString() }) + "\n";
        break;
      }

    } catch (err) {
      console.error("Error fetching job status:", err);
      // Retry after delay instead of breaking
    }

    await new Promise(resolve => setTimeout(resolve, 3000));
  }
}


export async function POST(request: NextRequest) {
  const { modelType, learningRate, batchSize, numSamples } = await request.json();
  if (!modelType || learningRate === undefined || !batchSize || !numSamples)
    return NextResponse.json({ error: "Missing hyperparameters" }, { status: 400 });
  const job = await submitAzureMLJob({ modelType, learningRate, batchSize, numSamples });
  const encoder = new TextEncoder();

const stream = new ReadableStream({
  async start(controller) {

    // ✅ heartbeat every 2 seconds
    const heartbeat = setInterval(() => {
      controller.enqueue(
        encoder.encode(JSON.stringify({ ping: Date.now() }) + "\n")
      );
    }, 2000);

    // ✅ initial event
    controller.enqueue(
      encoder.encode(JSON.stringify({
        jobId: job.jobId,
        status: "submitted",
        timestamp: new Date().toISOString(),
      }) + "\n")
    );

    try {
      for await (const chunk of streamAzureMLJobProgress(job.jobId)) {
        controller.enqueue(encoder.encode(chunk));
      }
    } finally {
      clearInterval(heartbeat);
      controller.close();
    }
  },
});

  return new NextResponse(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no" } });
}

export async function GET(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get("jobId");
  if (!jobId) return NextResponse.json({ error: "jobId parameter required" }, { status: 400 });
  const token = await getAzureMLAuthToken();
  const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID!;
  const resourceGroup = process.env.AZURE_RESOURCE_GROUP!;
  const workspaceName = process.env.AZURE_ML_WORKSPACE_NAME!;
  const apiUrl = `https://management.azure.com/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/jobs/${jobId}?api-version=2024-01-01-preview`;
  const response = await fetch(apiUrl, { method: "GET", headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" } });
  if (!response.ok) return NextResponse.json({ error: "Failed to get job status" }, { status: 500 });
  const job = await response.json();
  return NextResponse.json({ jobId, status: job.properties?.status || "unknown" });
}
