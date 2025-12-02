// app/api/train-custom/route.ts
import { NextRequest, NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import { BlobServiceClient } from "@azure/storage-blob";

interface HyperParameters {
  hiddenUnits: number;
  learningRate: number;
  epochs: number;
  csvFileName: string;
  dependentVariable: string;
}

async function uploadToBlob(content: string, blobName: string): Promise<string> {
  const connectionString = process.env.AZURE_STORAGE_CONNECTION_STRING!;
  const containerName = "training-inputs";
  
  const blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
  const containerClient = blobServiceClient.getContainerClient(containerName);
  
  // Create container if it doesn't exist
  await containerClient.createIfNotExists({ access: "blob" });
  
  const blockBlobClient = containerClient.getBlockBlobClient(blobName);
  await blockBlobClient.upload(content, content.length);
  
  return blockBlobClient.url;
}

async function getAzureMLAuthToken(): Promise<string> {
  const tenantId = process.env.AZURE_TENANT_ID!;
  const tokenUrl = `https://login.microsoftonline.com/${tenantId}/oauth2/v2.0/token`;
  const params = new URLSearchParams({ 
    client_id: process.env.AZURE_CLIENT_ID!, 
    client_secret: process.env.AZURE_CLIENT_SECRET!, 
    grant_type: "client_credentials", 
    scope: "https://management.azure.com/.default" 
  });
  const response = await fetch(tokenUrl, { 
    method: "POST", 
    headers: { "Content-Type": "application/x-www-form-urlencoded" }, 
    body: params 
  });
  if (!response.ok) throw new Error("Azure auth failed");
  return (await response.json()).access_token;
}

async function submitAzureMLJob(params: HyperParameters, csvContent: string) {
  const token = await getAzureMLAuthToken();
  const scriptPath = path.join(process.cwd(), "scripts", "train_spam.py");
  const scriptContent = fs.readFileSync(scriptPath, "utf-8");
  
  const timestamp = Date.now();
  const jobId = `bert-custom-${timestamp}`;
  
  // Upload script and CSV to blob storage
  console.log("Uploading training script to blob storage...");
  const scriptUrl = await uploadToBlob(scriptContent, `${jobId}/train_spam.py`);
  
  console.log("Uploading CSV to blob storage...");
  const csvUrl = await uploadToBlob(csvContent, `${jobId}/${params.csvFileName}`);
  
  const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID!;
  const resourceGroup = process.env.AZURE_RESOURCE_GROUP!;
  const workspaceName = process.env.AZURE_ML_WORKSPACE_NAME!;
  const apiUrl = `https://management.azure.com/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/jobs/${jobId}?api-version=2024-01-01-preview`;
  
  // Create simple command that downloads files from blob storage
  const command = [
    `wget -O train_spam.py "${scriptUrl}"`,
    `wget -O ${params.csvFileName} "${csvUrl}"`,
    `pip install transformers scikit-learn pandas azure-storage-blob tqdm --break-system-packages`,
    `python train_spam.py --hidden-units ${params.hiddenUnits} --learning-rate ${params.learningRate} --epochs ${params.epochs} --batch-size 32 --csv-path ${params.csvFileName} --dependent-var "${params.dependentVariable}"`
  ].join(" && ");
  
  const jobConfig = { 
    properties: { 
      jobType: "Command", 
      command: command,
      environmentId: `/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/environments/AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu/versions/10`, 
      computeId: `/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/${workspaceName}/computes/cpu-cluster`, 
      experimentName: "bert-custom-classification", 
      displayName: `DistilBERT Custom - ${params.dependentVariable} - ${new Date().toISOString()}`, 
      description: `DistilBERT classification on ${params.csvFileName}`, 
      environmentVariables: { 
        AZURE_STORAGE_CONNECTION_STRING: process.env.AZURE_STORAGE_CONNECTION_STRING || "" 
      } 
    } 
  };
  
  console.log("Submitting job to Azure ML...");
  const response = await fetch(apiUrl, { 
    method: "PUT", 
    headers: { 
      Authorization: `Bearer ${token}`, 
      "Content-Type": "application/json" 
    }, 
    body: JSON.stringify(jobConfig) 
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error("Job submission failed:", errorText);
    throw new Error(`Job submission failed: ${errorText}`);
  }
  
  console.log(`Job submitted successfully: ${jobId}`);
  return { jobId, status: "submitted", scriptUrl, csvUrl };
}

async function getBlobUrls(jobId: string, maxRetries: number = 40, delayMs: number = 5000) {
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
      // Manifest not ready yet
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
    const response = await fetch(apiUrl, { 
      method: "GET", 
      headers: { 
        Authorization: `Bearer ${token}`, 
        "Content-Type": "application/json" 
      } 
    });
    
    if (!response.ok) break;
    
    const job = await response.json();
    const status = job.properties?.status || "unknown";
    const errorDetails = job.properties?.error?.message || null;
    
    if (status !== previousStatus) {
      yield JSON.stringify({ 
        jobId, 
        status: status.toLowerCase(), 
        message: errorDetails || undefined,
        timestamp: new Date().toISOString() 
      }) + "\n";
      previousStatus = status;
    }
    
    if (status === "Completed") {
      // Stream a "waiting for outputs" message
      yield JSON.stringify({ 
        jobId, 
        status: "waiting_for_outputs", 
        message: "Job completed, waiting for output files to be uploaded to blob storage...",
        timestamp: new Date().toISOString() 
      }) + "\n";
      
      // Now wait for blob URLs with retry logic
      const urls = await getBlobUrls(jobId);
      
      // Fetch metrics if available
      let metrics = null;
      if (urls?.metrics_url) {
        try {
          const metricsResponse = await fetch(urls.metrics_url);
          if (metricsResponse.ok) {
            metrics = await metricsResponse.json();
          }
        } catch (e) {
          console.error("Failed to fetch metrics:", e);
        }
      }
      
      yield JSON.stringify({ 
        jobId, 
        status: "completed", 
        portalUrl: urls?.portal_url || null,
        results: {
          final_train_accuracy: metrics?.final_train_accuracy || null,
          final_val_accuracy: metrics?.final_val_accuracy || null,
          final_train_loss: metrics?.final_train_loss || null,
          final_val_loss: metrics?.final_val_loss || null,
          training_history: metrics?.training_history || null,
          model_url: urls?.model_url || null,
          results_url: urls?.results_url || null,
          metrics_url: urls?.metrics_url || null,
          manifest_url: urls?.manifest_url || null
        },
        timestamp: new Date().toISOString() 
      }) + "\n";
      break;
    }
    
    if (status === "Failed" || status === "Canceled") {
      yield JSON.stringify({ 
        jobId, 
        status: "error", 
        error: `Job ${status.toLowerCase()}${errorDetails ? ': ' + errorDetails : ''}`, 
        timestamp: new Date().toISOString() 
      }) + "\n";
      break;
    }
    
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    const file = formData.get("file") as File;
    const dependentVariable = formData.get("dependentVariable") as string;
    const hiddenUnits = parseInt(formData.get("hiddenUnits") as string);
    const learningRate = parseFloat(formData.get("learningRate") as string);
    const epochs = parseInt(formData.get("epochs") as string);
    
    if (!file || !dependentVariable || !hiddenUnits || !learningRate || !epochs) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 });
    }
    
    // Read CSV content
    const csvContent = await file.text();
    
    // Submit job with CSV content
    const job = await submitAzureMLJob(
      { 
        hiddenUnits, 
        learningRate, 
        epochs, 
        csvFileName: file.name,
        dependentVariable 
      },
      csvContent
    );
    
    // Stream progress
    const stream = new ReadableStream({ 
      async start(controller) { 
        controller.enqueue(
          new TextEncoder().encode(
            JSON.stringify({ 
              jobId: job.jobId, 
              status: "submitted", 
              timestamp: new Date().toISOString() 
            }) + "\n"
          )
        ); 
        
        for await (const chunk of streamAzureMLJobProgress(job.jobId)) { 
          controller.enqueue(new TextEncoder().encode(chunk)); 
        } 
        
        controller.close(); 
      } 
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
  
  const response = await fetch(apiUrl, { 
    method: "GET", 
    headers: { 
      Authorization: `Bearer ${token}`, 
      "Content-Type": "application/json" 
    } 
  });
  
  if (!response.ok) return NextResponse.json({ error: "Failed to get job status" }, { status: 500 });
  
  const job = await response.json();
  return NextResponse.json({ jobId, status: job.properties?.status || "unknown" });
}