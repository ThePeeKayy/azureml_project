// Azure configuration and SDK helpers
export const azureConfig = {
  // Azure ML Workspace
  subscriptionId: process.env.AZURE_SUBSCRIPTION_ID || "",
  resourceGroup: process.env.AZURE_RESOURCE_GROUP || "",
  workspaceName: process.env.AZURE_ML_WORKSPACE_NAME || "",

  // Azure Cognitive Services (Free tier available)
  cognitiveServicesKey: process.env.AZURE_COGNITIVE_SERVICES_KEY || "",
  cognitiveServicesEndpoint: process.env.AZURE_COGNITIVE_SERVICES_ENDPOINT || "",

  // Azure Storage (for model artifacts)
  storageAccountName: process.env.AZURE_STORAGE_ACCOUNT_NAME || "",
  storageAccountKey: process.env.AZURE_STORAGE_ACCOUNT_KEY || "",

  // Azure Functions (consumption plan - very cheap)
  functionAppUrl: process.env.AZURE_FUNCTION_APP_URL || "",
  functionAppKey: process.env.AZURE_FUNCTION_APP_KEY || "",
}

export function validateAzureConfig() {
  const missing = []

  if (!azureConfig.subscriptionId) missing.push("AZURE_SUBSCRIPTION_ID")
  if (!azureConfig.resourceGroup) missing.push("AZURE_RESOURCE_GROUP")
  if (!azureConfig.cognitiveServicesKey) missing.push("AZURE_COGNITIVE_SERVICES_KEY")

  return {
    isValid: missing.length === 0,
    missing,
  }
}
