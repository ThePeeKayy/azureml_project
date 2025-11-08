"use client"

import { Navigation } from "@/components/navigation"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { useState, useEffect } from "react"
import { Play, Download, Database, Zap, AlertCircle, CheckCircle, Clock, FileJson, FileCode, BarChart3 } from "lucide-react"

interface TrainingUpdate {
  portalUrl: string | URL | undefined
  jobId: string
  status: "submitted" | "notstarted" | "queued" | "preparing" | "starting" | "running" | "completed" | "error" | "waiting_for_outputs"
  timestamp: string
  modelUrl?: string
  resultsUrl?: string
  metricsUrl?: string
  results?: { 
    model_url?: string
    results_url?: string
    metrics_url?: string
    manifest_url?: string
  }
  error?: string
  message?: string
}

export default function Page() {
  const [modelType, setModelType] = useState<"simple" | "traffic">("simple")
  const [learningRate, setLearningRate] = useState([0.001])
  const [batchSize, setBatchSize] = useState([64])
  const [numSamples, setNumSamples] = useState([10000])
  const [isTraining, setIsTraining] = useState(false)
  const [trainingData, setTrainingData] = useState<TrainingUpdate | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string>("")

  const handleTrain = async () => {
    setIsTraining(true)
    setError(null)
    setTrainingData(null)
    setCurrentStatus("")

    try {
      const response = await fetch("/api/train-cnn", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          modelType,
          learningRate: learningRate[0],
          batchSize: batchSize[0],
          numSamples: numSamples[0],
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to start training")
      }

      if (!response.body) {
        throw new Error("No response body")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          console.log("Stream ended")
          break
        }

        const chunk = decoder.decode(value)
        console.log("Received chunk:", chunk)
        const lines = chunk.split("\n").filter((line) => line.trim())

        for (const line of lines) {
          try {
            const update: TrainingUpdate = JSON.parse(line)
            console.log("Parsed update:", update)

            setJobId(update.jobId)
            setCurrentStatus(update.status)
            setTrainingData(update)

            if (update.status === "completed") {
              console.log("Training completed!")
              console.log("Results:", update.results)
              setIsTraining(false)
            } else if (update.status === "error") {
              console.error("Training error:", update.error)
              setError(update.error || "Training failed")
              setIsTraining(false)
            }

          } catch (e) {
            console.error("Error parsing update:", e, "Line:", line)
          }
        }
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Training failed"
      setError(errorMsg)
      setIsTraining(false)
    }
  }

  const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { label: string; color: string }> = {
      submitted: { label: "Submitted", color: "bg-blue-500/20 text-blue-400 border-blue-500/30" },
      notstarted: { label: "Not Started", color: "bg-slate-500/20 text-slate-400 border-slate-500/30" },
      queued: { label: "Queued", color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30" },
      preparing: { label: "Preparing", color: "bg-orange-500/20 text-orange-400 border-orange-500/30" },
      starting: { label: "Starting", color: "bg-orange-500/20 text-orange-400 border-orange-500/30" },
      running: { label: "Running", color: "bg-purple-500/20 text-purple-400 border-purple-500/30" },
      waiting_for_outputs: { label: "Processing Outputs", color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30" },
      completed: { label: "Completed", color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" },
      error: { label: "Error", color: "bg-red-500/20 text-red-400 border-red-500/30" },
    }
    
    const statusInfo = statusMap[status] || statusMap.notstarted
    return (
      <Badge variant="outline" className={`${statusInfo.color}`}>
        {statusInfo.label}
      </Badge>
    )
  }

  return (
    <div className="min-h-screen bg-slate-950">
      <Navigation />

      <div className="pt-24 pb-12 px-4">
        <div className="container mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-8">
            
            <h1 className="text-4xl font-bold mb-3 text-white">Fast CNN Training</h1>
            <p className="text-lg text-slate-400 max-w-3xl leading-relaxed">
              Train efficient CNN models on CIFAR-10. Choose MobileNetV3-Small (simple, ~1.5M params) 
              or ShuffleNetV2 (traffic, ~2.3M params) for fast training on CPU.
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-950/50 border border-red-800 rounded-lg flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400">Training Error</p>
                <p className="text-sm text-red-300">{error}</p>
              </div>
            </div>
          )}

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Hyperparameters Panel */}
            <Card className="p-6 bg-slate-900 border-slate-800">
              <h2 className="text-xl font-semibold mb-6 text-white">Hyperparameters</h2>

              <div className="space-y-6">
                {/* Model Type Selection */}
                <div>
                  <Label className="text-slate-300 mb-3 block">Model Architecture</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant={"default"}
                      className={modelType === "simple" ? "" : "bg-gray-700"}
                      onClick={() => setModelType("simple")}
                      disabled={isTraining || (currentStatus !== "" && currentStatus !== "completed" && currentStatus !== "error")}

                    >
                      Simple
                    </Button>
                    <Button
                      variant={"default"}
                      className={modelType === "traffic" ? "" : "bg-gray-700"}
                      onClick={() => setModelType("traffic")}
                      disabled={isTraining || (currentStatus !== "" && currentStatus !== "completed" && currentStatus !== "error")}

                    >
                      Traffic
                    </Button>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    {modelType === "simple" 
                      ? "MobileNetV3-Small (~1.5M params, optimized for speed)"
                      : "ShuffleNetV2 (~2.3M params, very fast inference)"}
                  </p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-slate-300">Training Samples</Label>
                    <span className="text-sm font-mono text-blue-400">{numSamples[0].toLocaleString()}</span>
                  </div>
                  <Slider
                    min={5000}
                    max={50000}
                    step={5000}
                    value={numSamples}
                    onValueChange={setNumSamples}
                    disabled={isTraining || (currentStatus !== "" && currentStatus !== "completed" && currentStatus !== "error")}

                  />
                  <p className="text-xs text-slate-500 mt-2">
                    CIFAR-10 training images
                  </p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-slate-300">Learning Rate</Label>
                    <span className="text-sm font-mono text-slate-400">{learningRate[0].toFixed(4)}</span>
                  </div>
                  <Slider
                    min={0.0001}
                    max={0.01}
                    step={0.0001}
                    value={learningRate}
                    onValueChange={setLearningRate}
                    disabled={isTraining || (currentStatus !== "" && currentStatus !== "completed" && currentStatus !== "error")}

                  />
                  <p className="text-xs text-slate-500 mt-2">Adam optimizer with ReduceLROnPlateau scheduler</p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-slate-300">Batch Size</Label>
                    <span className="text-sm font-mono text-slate-400">{batchSize[0]}</span>
                  </div>
                  <Slider
                    min={32}
                    max={128}
                    step={32}
                    value={batchSize}
                    onValueChange={setBatchSize}
                    disabled={isTraining || (currentStatus !== "" && currentStatus !== "completed" && currentStatus !== "error")}

                  />
                  <p className="text-xs text-slate-500 mt-2">Training batch size (32-128)</p>
                </div>

                <div className="pt-4 space-y-3">
                  <Button
                    onClick={handleTrain}
                    disabled={isTraining || (currentStatus !== "" && currentStatus !== "completed" && currentStatus !== "error")}

                    className="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    {isTraining ? (
                      <>
                        <Zap className="h-4 w-4 mr-2 animate-pulse" />
                        Training...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Start Training
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Azure Config */}
              <div className="mt-6 pt-6 border-t border-slate-800">
                <h3 className="text-sm font-semibold mb-3 text-slate-300">Configuration</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Compute:</span>
                    <span className="font-mono text-xs text-slate-400">Azure CPU Cluster</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Model:</span>
                    <span className="font-mono text-xs text-purple-400">
                      {modelType === "simple" ? "MobileNetV3-Small" : "ShuffleNetV2"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Parameters:</span>
                    <span className="font-mono text-xs text-slate-400">
                      {modelType === "simple" ? "~1.5M" : "~2.3M"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Dataset:</span>
                    <span className="font-mono text-xs text-emerald-400">CIFAR-10</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Epochs:</span>
                    <span className="font-mono text-xs text-slate-400">2</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Samples:</span>
                    <span className="font-mono text-xs text-blue-400">{numSamples[0].toLocaleString()}</span>
                  </div>
                  {jobId && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">Job ID:</span>
                      <span className="font-mono text-xs text-blue-400 truncate max-w-[120px]" title={jobId}>
                        {jobId.slice(-12)}
                      </span>
                    </div>
                  )}
                  {currentStatus && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-500">Status:</span>
                      {getStatusBadge(currentStatus)}
                    </div>
                  )}
                </div>
              </div>
            </Card>

            {/* Training & Results */}
            <div className="lg:col-span-2 space-y-6">
              {/* Training Status */}
              {isTraining && currentStatus && (
                <Card className="p-6 bg-slate-900 border-slate-800">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-white">Training Status</h2>
                    {getStatusBadge(currentStatus)}
                  </div>
                  <div className="flex items-center gap-3 text-slate-400">
                    <Clock className="h-5 w-5 animate-pulse" />
                    <span>
                      {currentStatus === "waiting_for_outputs" 
                        ? trainingData?.message || "Processing outputs..."
                        : `Job ${jobId?.slice(-12)} is ${currentStatus}...`}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    Training with {numSamples[0].toLocaleString()} samples on Azure ML CPU cluster
                  </p>
                </Card>
              )}

              {/* Completion Results */}
              {trainingData?.status === "completed" && trainingData.results && (
                <Card className="p-6 bg-blue-950/50 border-gray-800 space-y-4">
                  <div className="flex items-start gap-3">
                    <CheckCircle className="h-6 w-6 text-blue-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-blue-400 mb-2">Training Complete!</h3>
                      <p className="text-sm text-blue-300 mb-4">
                        Your model has been trained and all outputs are available for download below.
                      </p>
                    </div>
                  </div>

                  {/* Download Buttons - All 4 Links */}
                  <div className="space-y-3">
                    {trainingData.results?.model_url && (
                      <Button 
                        variant="outline" 
                        className="p-5 w-full justify-start bg-slate-900/50 hover:bg-slate-900 border-slate-700" 
                        onClick={() => window.open(trainingData.results?.model_url, "_blank")}
                      >
                        <FileCode className="h-4 w-4 mr-2 text-purple-400" />
                        <div className="flex-1 text-left">
                          <div className="font-medium text-white">Model Weights (.pth)</div>
                          <div className="text-xs text-slate-400">PyTorch trained model checkpoint</div>
                        </div>
                        <Download className="h-4 w-4 ml-2 text-slate-500" />
                      </Button>
                    )}
                    
                    {trainingData.results?.results_url && (
                      <Button 
                        variant="outline" 
                        className="p-5 w-full justify-start bg-slate-900/50 hover:bg-slate-900 border-slate-700" 
                        onClick={() => window.open(trainingData.results?.results_url, "_blank")}
                      >
                        <FileJson className="h-4 w-4 mr-2 text-blue-400" />
                        <div className="flex-1 text-left">
                          <div className="font-medium text-white">Training Results (.json)</div>
                          <div className="text-xs text-slate-400">Complete training history and metadata</div>
                        </div>
                        <Download className="h-4 w-4 ml-2 text-slate-500" />
                      </Button>
                    )}
                    
                    {trainingData.results?.metrics_url && (
                      <Button 
                        variant="outline" 
                        className="p-5 w-full justify-start bg-slate-900/50 hover:bg-slate-900 border-slate-700" 
                        onClick={() => window.open(trainingData.results?.metrics_url, "_blank")}
                      >
                        <BarChart3 className="h-4 w-4 mr-2 text-emerald-400" />
                        <div className="flex-1 text-left">
                          <div className="font-medium text-white">Metrics (.json)</div>
                          <div className="text-xs text-slate-400">Loss and accuracy per epoch</div>
                        </div>
                        <Download className="h-4 w-4 ml-2 text-slate-500" />
                      </Button>
                    )}
                    
                    {trainingData.results?.manifest_url && (
                      <Button 
                        variant="outline" 
                        className="p-5 w-full justify-start bg-slate-900/50 hover:bg-slate-900 border-slate-700" 
                        onClick={() => window.open(trainingData.results?.manifest_url, "_blank")}
                      >
                        <Database className="h-4 w-4 mr-2 text-cyan-400" />
                        <div className="flex-1 text-left">
                          <div className="font-medium text-white">Manifest (.json)</div>
                          <div className="text-xs text-slate-400">Job metadata and output file registry</div>
                        </div>
                        <Download className="h-4 w-4 ml-2 text-slate-500" />
                      </Button>
                    )}
                  </div>

                 
                </Card>
              )}

              {/* CIFAR-10 Info */}
              <Card className="p-6 bg-slate-900 border-slate-800">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-white">CIFAR-10 Dataset</h2>
                </div>
                <p className="text-sm text-slate-400 mb-4">
                  10 classes of 32×32 color images with data augmentation (random flip, crop)
                </p>
                <div className="grid grid-cols-5 gap-2 text-xs">
                  {['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'].map((cls) => (
                    <div key={cls} className="bg-slate-800 rounded p-2 text-center">
                      <span className="text-slate-300">{cls}</span>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Model Architecture */}
              <Card className="p-6 bg-slate-900 border-slate-800">
                <h2 className="text-xl font-semibold mb-4 text-white">Model Architecture</h2>
                {modelType === "simple" ? (
                  <div className="space-y-3">
                    <div className="p-3 bg-slate-800 rounded-lg">
                      <p className="text-sm font-semibold text-slate-300 mb-2">MobileNetV3-Small</p>
                      <p className="text-xs text-slate-400">
                        Efficient architecture with inverted residuals, squeeze-and-excitation blocks, 
                        and hard-swish activations. Optimized for mobile and edge devices.
                      </p>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">Input</span>
                      <span className="text-xs font-mono text-slate-500">32×32×3</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">MBConv Blocks</span>
                      <span className="text-xs font-mono text-slate-500">11 layers</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">Classifier</span>
                      <span className="text-xs font-mono text-slate-500">→ 10 classes</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">Total Parameters</span>
                      <span className="text-xs font-mono text-blue-400">~1.5M</span>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="p-3 bg-slate-800 rounded-lg">
                      <p className="text-sm font-semibold text-slate-300 mb-2">ShuffleNetV2 x1.0</p>
                      <p className="text-xs text-slate-400">
                        Highly efficient architecture using channel shuffle operations for fast inference. 
                        Designed for real-time applications with minimal computational cost.
                      </p>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">Input</span>
                      <span className="text-xs font-mono text-slate-500">32×32×3</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">Shuffle Units</span>
                      <span className="text-xs font-mono text-slate-500">4 stages</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">Classifier</span>
                      <span className="text-xs font-mono text-slate-500">→ 10 classes</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                      <span className="text-sm text-slate-300">Total Parameters</span>
                      <span className="text-xs font-mono text-blue-400">~2.3M</span>
                    </div>
                  </div>
                )}
                
                <div className="mt-4 p-3 bg-blue-950/30 border border-blue-800/30 rounded-lg">
                  <p className="text-xs text-blue-300">
                    <strong>Optimizer:</strong> Adam with ReduceLROnPlateau scheduler (reduces LR on plateau by 0.5×)
                  </p>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}