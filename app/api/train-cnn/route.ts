// app/api/train-cnn/route.ts
import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import { BlobServiceClient } from "@azure/storage-blob";

interface HyperParameters {
  modelType: string;
  learningRate: number;
  batchSize: number;
  epochs: number;
  datasetFileName: string;
}

async function uploadToBlob(content: Buffer | string, blobName: string): Promise<string> {
  const connectionString = process.env.AZURE_STORAGE_CONNECTION_STRING!;
  const containerName = "training-inputs";
  
  const blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
  const containerClient = blobServiceClient.getContainerClient(containerName);
  await containerClient.createIfNotExists({ access: "blob" });
  
  const blockBlobClient = containerClient.getBlockBlobClient(blobName);
  if (typeof content === 'string') {
    await blockBlobClient.upload(content, content.length);
  } else {
    await blockBlobClient.upload(content, content.length);
  }
  
  return blockBlobClient.url;
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
  const response = await fetch(tokenUrl, { 
    method: "POST", 
    headers: { "Content-Type": "application/x-www-form-urlencoded" }, 
    body: params 
  });
  if (!response.ok) throw new Error("Azure auth failed");
  return (await response.json()).access_token;
}

async function submitAzureMLJob(params: HyperParameters, datasetBuffer: Buffer) {
  const token = await getAzureMLAuthToken();
  const scriptPath = path.join(process.cwd(), "scripts", "train_cnn.py");
  const scriptContent = fs.readFileSync(scriptPath, "utf-8");
  const jobId = `cnn-training-${Date.now()}`;
  const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID!;
  const resourceGroup = process.env.AZURE_RESOURCE_GROUP!;
  const workspaceName = process.env.AZURE_ML_WORKSPACE_NAME!;
  const apiUrl = `https://management.azure.com/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/jobs/${jobId}?api-version=2024-01-01-preview`;

  // Upload script and dataset to blob storage
  console.log("Uploading training script to blob storage...");
  const scriptUrl = await uploadToBlob(scriptContent, `${jobId}/train_cnn.py`);
  
  console.log("Uploading dataset to blob storage...");
  const datasetUrl = await uploadToBlob(datasetBuffer, `${jobId}/${params.datasetFileName}`);

  // Build command
  const command = [
    `wget -O train_cnn.py "${scriptUrl}"`,
    `wget -O ${params.datasetFileName} "${datasetUrl}"`,
    `pip install torch torchvision pillow azure-storage-blob tqdm --break-system-packages`,
    `python train_cnn.py --model-type ${params.modelType} --lr ${params.learningRate} --batch-size ${params.batchSize} --epochs ${params.epochs} --dataset-path ${params.datasetFileName}`
  ].join(" && ");

  const jobConfig = {
    properties: {
      jobType: "Command",
      command: command,
      environmentId: `/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/environments/AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu/versions/10`,
      computeId: `/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/computes/cpu-cluster`,
      experimentName: "cnn-training",
      displayName: `Fast CNN Training - ${params.modelType} - ${new Date().toISOString()}`,
      description: `Fast CNN training on custom dataset`,
      environmentVariables: { AZURE_STORAGE_CONNECTION_STRING: process.env.AZURE_STORAGE_CONNECTION_STRING || "" },
    },
  };

  const response = await fetch(apiUrl, { 
    method: "PUT", 
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }, 
    body: JSON.stringify(jobConfig) 
  });

  if (!response.ok) throw new Error("Job submission failed");
  return { jobId, status: "submitted" };
}

async function getBlobUrls(jobId: string, maxRetries: number = 20, delayMs: number = 5000) {
  const connectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
  if (!connectionString) return null;
  const accountNameMatch = connectionString.match(/AccountName=([^;]+)/);
  if (!accountNameMatch) return null;
  const accountName = accountNameMatch[1];
  const baseUrl = `https://${accountName}.blob.core.windows.net/model-outputs/${jobId}`;
  const manifestUrl = `${baseUrl}/manifest.json`;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const manifestResponse = await fetch(manifestUrl);
      if (manifestResponse.ok) {
        const manifest = await manifestResponse.json();
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
    await new Promise(resolve => setTimeout(resolve, delayMs));
  }

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
      if (!response.ok) throw new Error("Failed to fetch job status");

      const job = await response.json();
      const status = job.properties?.status || "unknown";

      if (status !== previousStatus) {
        yield JSON.stringify({ jobId, status: status.toLowerCase(), timestamp: new Date().toISOString() }) + "\n";
        previousStatus = status;
      }

      if (status === "Completed") {
        yield JSON.stringify({ jobId, status: "waiting_for_outputs", timestamp: new Date().toISOString() }) + "\n";
        const urls = await getBlobUrls(jobId);
        yield JSON.stringify({ jobId, status: "completed", results: urls, timestamp: new Date().toISOString() }) + "\n";
        break;
      }

      if (status === "Failed" || status === "Canceled") {
        yield JSON.stringify({ jobId, status: "error", error: `Job ${status.toLowerCase()}`, timestamp: new Date().toISOString() }) + "\n";
        break;
      }

    } catch (err) {
      console.error("Error fetching job status:", err);
    }

    await new Promise(resolve => setTimeout(resolve, 3000));
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    const file = formData.get("file") as File;
    const modelType = formData.get("modelType") as string;
    const learningRate = parseFloat(formData.get("learningRate") as string);
    const batchSize = parseInt(formData.get("batchSize") as string);
    const epochs = parseInt(formData.get("epochs") as string);
    
    if (!file || !modelType || !learningRate || !batchSize || !epochs) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 });
    }

    const datasetBuffer = Buffer.from(await file.arrayBuffer());

    const job = await submitAzureMLJob({ 
      modelType, 
      learningRate, 
      batchSize, 
      epochs,
      datasetFileName: file.name
    }, datasetBuffer);
    
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      async start(controller) {
        controller.enqueue(encoder.encode(JSON.stringify({
          jobId: job.jobId,
          status: "submitted",
          timestamp: new Date().toISOString(),
        }) + "\n"));

        try {
          for await (const chunk of streamAzureMLJobProgress(job.jobId)) {
            controller.enqueue(encoder.encode(chunk));
          }
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
      }
    });
  } catch (error) {
    console.error("Error in POST:", error);
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : "Unknown error" 
    }, { status: 500 });
  }
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