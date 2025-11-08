import { NextResponse } from "next/server"
import { azureConfig } from "@/lib/azure-config"

export async function GET() {
  return NextResponse.json({
    subscriptionId: !!azureConfig.subscriptionId,
    resourceGroup: !!azureConfig.resourceGroup,
    cognitiveServices: !!(azureConfig.cognitiveServicesKey && azureConfig.cognitiveServicesEndpoint),
    workspace: !!azureConfig.workspaceName,
    storage: !!(azureConfig.storageAccountName && azureConfig.storageAccountKey),
    functions: !!(azureConfig.functionAppUrl && azureConfig.functionAppKey),
  })
}
