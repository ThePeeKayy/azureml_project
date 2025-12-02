"use client"

import type React from "react"

import { Navigation } from "@/components/navigation"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { useState } from "react"
import {
  Play,
  AlertCircle,
  CheckCircle,
  Download,
  FileCode,
  FileJson,
  Database,
  Clock,
  FolderOpen,
  X,
  ImageIcon,
} from "lucide-react"

interface TrainingUpdate {
  portalUrl?: string | URL
  jobId: string
  status: string
  timestamp: string
  results?: {
    final_train_accuracy?: number | null
    final_val_accuracy?: number | null
    final_train_loss?: number | null
    final_val_loss?: number | null
    training_history?: {
      train_losses: number[]
      train_accuracies: number[]
      val_losses: number[]
      val_accuracies: number[]
    }
    model_url?: string
    results_url?: string
    metrics_url?: string
    manifest_url?: string
  }
  error?: string
  message?: string
}

export default function CustomImageTrainingPage() {
  // Model parameters
  const [modelType, setModelType] = useState<"mobilenet" | "shufflenet">("mobilenet")
  const [learningRate, setLearningRate] = useState([0.001])
  const [batchSize, setBatchSize] = useState([64])
  const [epochs, setEpochs] = useState([2])

  // Dataset upload state
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<{ classes: string[]; totalImages: number } | null>(null)

  // Training state
  const [isTraining, setIsTraining] = useState(false)
  const [trainingData, setTrainingData] = useState<TrainingUpdate | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string>("")

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith(".zip")) {
      setError("Please upload a ZIP file containing your image dataset")
      return
    }

    setDatasetFile(file)
    setError(null)

    // Try to extract dataset info from the zip
    try {
      // We'll validate on the backend, but show basic info here
      setDatasetInfo({
        classes: [],
        totalImages: 0,
      })
    } catch (err) {
      setError("Failed to read dataset file")
    }
  }

  const clearDataset = () => {
    setDatasetFile(null)
    setDatasetInfo(null)
  }

  const loadSampleData = async () => {
    try {
      const response = await fetch("/pistachio.zip")
      if (!response.ok) throw new Error("Failed to load sample data")

      const blob = await response.blob()
      const file = new File([blob], "pistachio.zip", { type: "application/zip" })
      setDatasetFile(file)

      setDatasetInfo({
        classes: ["class1", "class2"],
        totalImages: 100,
      })
      setError(null)
    } catch (err) {
      setError("Failed to load sample data. Make sure sample-dataset.zip is in the public folder.")
    }
  }

  const handleTrain = async () => {
    if (!datasetFile) {
      setError("Please upload a dataset ZIP file")
      return
    }

    setIsTraining(true)
    setError(null)
    setTrainingData(null)
    setCurrentStatus("")

    try {
      const formData = new FormData()
      formData.append("file", datasetFile)
      formData.append("modelType", modelType)
      formData.append("learningRate", learningRate[0].toString())
      formData.append("batchSize", batchSize[0].toString())
      formData.append("epochs", epochs[0].toString())

      const response = await fetch("/api/train-cnn", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Failed to start training")
      if (!response.body) throw new Error("No response body")

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split("\n").filter((line) => line.trim())

        for (const line of lines) {
          try {
            const update: TrainingUpdate = JSON.parse(line)
            console.log("Parsed update:", update)
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
            console.error("Error parsing update:", e)
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
      <div className="pt-16 md:pt-24 pb-8 md:pb-12 px-3 md:px-4">
        <div className="container mx-auto max-w-7xl">
          <div className="mb-6 md:mb-8">
            <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-3 text-white">
              Custom Image Classification Training
            </h1>
            <p className="text-base md:text-lg text-slate-400">
              Upload your image dataset and train a custom CNN model using MobileNetV3 or ShuffleNetV2 on Azure ML
            </p>
          </div>

          {error && (
            <div className="mb-4 md:mb-6 p-3 md:p-4 bg-red-950/50 border border-red-800 rounded-lg flex items-start gap-2 md:gap-3">
              <AlertCircle className="h-4 w-4 md:h-5 md:w-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-xs md:text-sm font-medium text-red-400">Error</p>
                <p className="text-xs md:text-sm text-red-300">{error}</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
            {/* Hyperparameters Panel */}
            <div className="space-y-4 md:space-y-6">
              <Card className="p-4 md:p-6 bg-slate-900 border-slate-800">
                <h2 className="text-lg md:text-xl font-semibold mb-4 md:mb-6 text-white">Hyperparameters</h2>
                <div className="space-y-4 md:space-y-6">
                  {/* Model Type Selection */}
                  <div>
                    <Label className="text-sm md:text-base text-slate-300 mb-2 md:mb-3 block">Model Architecture</Label>
                    <div className="grid grid-cols-2 gap-2">
                      <Button
                        onClick={() => setModelType("mobilenet")}
                        disabled={isTraining}
                        className={
                          modelType === "mobilenet"
                            ? "text-sm md:text-base"
                            : "bg-slate-800 hover:bg-slate-700 text-sm md:text-base"
                        }
                      >
                        MobileNet
                      </Button>
                      <Button
                        onClick={() => setModelType("shufflenet")}
                        disabled={isTraining}
                        className={
                          modelType === "shufflenet"
                            ? "text-sm md:text-base"
                            : "bg-slate-800 hover:bg-slate-700 text-sm md:text-base"
                        }
                      >
                        ShuffleNet
                      </Button>
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                      {modelType === "mobilenet"
                        ? "MobileNetV3-Small (~1.5M params, fastest)"
                        : "ShuffleNetV2 (~2.3M params, very fast)"}
                    </p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2 md:mb-3">
                      <Label className="text-sm md:text-base text-slate-300">Learning Rate</Label>
                      <span className="text-xs md:text-sm font-mono text-slate-400">{learningRate[0].toFixed(4)}</span>
                    </div>
                    <Slider
                      min={0.0001}
                      max={0.01}
                      step={0.0001}
                      value={learningRate}
                      onValueChange={setLearningRate}
                      disabled={isTraining}
                    />
                    <p className="text-xs text-slate-500 mt-2">Adam optimizer with ReduceLROnPlateau scheduler</p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2 md:mb-3">
                      <Label className="text-sm md:text-base text-slate-300">Batch Size</Label>
                      <span className="text-xs md:text-sm font-mono text-slate-400">{batchSize[0]}</span>
                    </div>
                    <Slider
                      min={32}
                      max={128}
                      step={32}
                      value={batchSize}
                      onValueChange={setBatchSize}
                      disabled={isTraining}
                    />
                    <p className="text-xs text-slate-500 mt-2">Training batch size (32-128)</p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2 md:mb-3">
                      <Label className="text-sm md:text-base text-slate-300">Training Epochs</Label>
                      <span className="text-xs md:text-sm font-mono text-slate-400">{epochs[0]}</span>
                    </div>
                    <Slider min={1} max={10} step={1} value={epochs} onValueChange={setEpochs} disabled={isTraining} />
                    <p className="text-xs text-slate-500 mt-2">Number of times to iterate over the dataset</p>
                  </div>

                  <Button
                    onClick={handleTrain}
                    disabled={isTraining || !datasetFile}
                    className="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    {isTraining ? (
                      <>
                        <Play className="h-4 w-4 mr-2 animate-pulse" />
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
              </Card>

              <Card className="p-4 md:p-6 bg-slate-900 border-slate-800">
                <h2 className="text-lg md:text-xl font-semibold mb-3 md:mb-4 text-white">Model Architecture</h2>
                {modelType === "mobilenet" ? (
                  <div className="space-y-2 md:space-y-3">
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">MobileNetV3-Small</span>
                      <span className="text-[10px] md:text-xs font-mono text-slate-500">11 layers</span>
                    </div>
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">Parameters</span>
                      <span className="text-[10px] md:text-xs font-mono text-blue-400">~1.5M</span>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2 md:space-y-3">
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">ShuffleNetV2 x1.0</span>
                      <span className="text-[10px] md:text-xs font-mono text-slate-500">4 stages</span>
                    </div>
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">Parameters</span>
                      <span className="text-[10px] md:text-xs font-mono text-blue-400">~2.3M</span>
                    </div>
                  </div>
                )}
                <div className="mt-3 md:mt-4 p-2 md:p-3 bg-blue-950/30 border border-blue-800/30 rounded-lg">
                  <p className="text-xs text-blue-300">
                    <strong>Optimizer:</strong> Adam with ReduceLROnPlateau scheduler
                  </p>
                </div>
              </Card>
            </div>

            {/* Dataset Upload and Results */}
            <div className="lg:col-span-2 space-y-4 md:space-y-6">
              <Card className="p-4 md:p-6 bg-slate-900 border-slate-800">
                <h2 className="text-lg md:text-xl font-semibold mb-3 md:mb-4 text-white">Upload Image Dataset</h2>

                {!datasetFile ? (
                  <div>
                    <div className="border-2 border-dashed border-slate-700 rounded-lg p-6 md:p-8 text-center hover:border-slate-600 transition-colors">
                      <input
                        type="file"
                        accept=".zip"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="dataset-upload"
                      />
                      <label htmlFor="dataset-upload" className="cursor-pointer">
                        <FolderOpen className="h-10 w-10 md:h-12 md:w-12 text-slate-500 mx-auto mb-3 md:mb-4" />
                        <p className="text-sm md:text-base text-slate-300 mb-1 md:mb-2">Click to upload ZIP file</p>
                        <p className="text-xs md:text-sm text-slate-500">ZIP file with folders for each class</p>
                      </label>
                    </div>

                    <div className="mt-3 md:mt-4 p-3 md:p-4 bg-slate-800 rounded-lg">
                      <p className="text-xs font-semibold text-slate-300 mb-2">Dataset Structure:</p>
                      <div className="overflow-x-auto">
                        <pre className="text-[10px] md:text-xs text-slate-400 font-mono whitespace-pre">
                          {`dataset.zip
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...`}
                        </pre>
                      </div>
                    </div>

                    <div className="mt-3 md:mt-4 flex items-center gap-2 md:gap-3">
                      <div className="flex-1 h-px bg-slate-700"></div>
                      <span className="text-xs text-slate-500">OR</span>
                      <div className="flex-1 h-px bg-slate-700"></div>
                    </div>

                    <Button
                      onClick={loadSampleData}
                      variant="outline"
                      className="w-full mt-3 md:mt-4 bg-slate-800 border-slate-700 hover:bg-slate-700 text-white text-sm md:text-base"
                    >
                      <ImageIcon className="h-4 w-4 mr-2 text-white" />
                      <span className="truncate">Load Sample Dataset (Pistachio dataset from Kaggle)</span>
                    </Button>
                  </div>
                ) : (
                  <div>
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-3 md:p-4 bg-slate-800 rounded-lg mb-3 md:mb-4 gap-2">
                      <div className="flex items-center gap-2 md:gap-3">
                        <FolderOpen className="h-4 w-4 md:h-5 md:w-5 text-blue-400 flex-shrink-0" />
                        <div>
                          <p className="text-xs md:text-sm font-medium text-white break-all">{datasetFile.name}</p>
                          <p className="text-[10px] md:text-xs text-slate-500">
                            {(datasetFile.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={clearDataset}
                        disabled={isTraining}
                        className="text-slate-400 hover:text-red-400 text-xs md:text-sm w-full sm:w-auto"
                      >
                        <X className="h-3 w-3 md:h-4 md:w-4" />
                        <span className="ml-1">Remove</span>
                      </Button>
                    </div>

                    <div className="p-3 md:p-4 bg-blue-950/30 border border-blue-800/30 rounded-lg">
                      <p className="text-xs md:text-sm text-blue-300">
                        ✓ Dataset uploaded successfully. Click "Start Training" to begin.
                      </p>
                    </div>
                  </div>
                )}
              </Card>

              {isTraining && currentStatus && (
                <Card className="p-4 md:p-6 bg-slate-900 border-slate-800">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-3 md:mb-4 gap-2">
                    <h2 className="text-lg md:text-xl font-semibold text-white">Training Status</h2>
                    {getStatusBadge(currentStatus)}
                  </div>
                  <div className="flex items-center gap-2 md:gap-3 text-slate-400">
                    <Clock className="h-4 w-4 md:h-5 md:w-5 animate-pulse flex-shrink-0" />
                    <span className="text-xs md:text-sm">
                      {currentStatus === "waiting_for_outputs"
                        ? trainingData?.message || "Processing outputs..."
                        : `Job is ${currentStatus}...`}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">Training your custom image classifier on Azure ML</p>
                </Card>
              )}

              {trainingData?.status === "completed" && trainingData.results && (
                <Card className="p-4 md:p-6 bg-gray-900 border-gray-800 space-y-3 md:space-y-4">
                  <div className="flex items-start gap-2 md:gap-3">
                    <CheckCircle className="h-5 w-5 md:h-6 md:w-6 text-blue-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <h3 className="text-base md:text-lg font-semibold text-blue-400 mb-2">Training Complete!</h3>

                      {/* Training Metrics */}
                      <div className="bg-slate-900/50 rounded-lg p-3 md:p-4 mb-3 md:mb-4 space-y-2">
                        {trainingData.results.final_train_accuracy && (
                          <p className="text-xs md:text-sm">
                            <span className="text-slate-500">Training Accuracy:</span>{" "}
                            <span className="font-mono text-emerald-300">
                              {(trainingData.results.final_train_accuracy * 100).toFixed(2)}%
                            </span>
                          </p>
                        )}
                        {trainingData.results.final_val_accuracy && (
                          <p className="text-xs md:text-sm">
                            <span className="text-slate-500">Validation Accuracy:</span>{" "}
                            <span className="font-mono text-blue-300">
                              {(trainingData.results.final_val_accuracy * 100).toFixed(2)}%
                            </span>
                          </p>
                        )}
                        {trainingData.results.final_train_loss && (
                          <p className="text-xs md:text-sm">
                            <span className="text-slate-500">Training Loss:</span>{" "}
                            <span className="font-mono text-orange-300">
                              {trainingData.results.final_train_loss.toFixed(4)}
                            </span>
                          </p>
                        )}
                        {trainingData.results.final_val_loss && (
                          <p className="text-xs md:text-sm">
                            <span className="text-slate-500">Validation Loss:</span>{" "}
                            <span className="font-mono text-purple-300">
                              {trainingData.results.final_val_loss.toFixed(4)}
                            </span>
                          </p>
                        )}
                      </div>

                      {/* Download Buttons */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 md:gap-3">
                        {trainingData.results.model_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.model_url, "_blank")}
                            variant="outline"
                            className="w-full bg-blue-950/30 border-blue-800/50 hover:bg-blue-900/30 text-blue-400 text-xs md:text-sm"
                          >
                            <Download className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Model (.pt)
                          </Button>
                        )}
                        {trainingData.results.results_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.results_url, "_blank")}
                            variant="outline"
                            className="w-full bg-emerald-950/30 border-emerald-800/50 hover:bg-emerald-900/30 text-emerald-400 text-xs md:text-sm"
                          >
                            <FileCode className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Results
                          </Button>
                        )}
                        {trainingData.results.metrics_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.metrics_url, "_blank")}
                            variant="outline"
                            className="w-full bg-purple-950/30 border-purple-800/50 hover:bg-purple-900/30 text-purple-400 text-xs md:text-sm"
                          >
                            <FileJson className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Metrics
                          </Button>
                        )}
                        {trainingData.results.manifest_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.manifest_url, "_blank")}
                            variant="outline"
                            className="w-full bg-orange-950/30 border-orange-800/50 hover:bg-orange-900/30 text-orange-400 text-xs md:text-sm"
                          >
                            <Database className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Manifest
                          </Button>
                        )}
                      </div>

                      {/* Training History Chart */}
                      {trainingData.results.training_history && (
                        <div className="mt-4 md:mt-6">
                          <h4 className="text-sm md:text-base font-semibold text-white mb-3 md:mb-4">
                            Training History
                          </h4>
                          {/* Placeholder for chart */}
                          <div className="bg-slate-800 rounded-lg p-4">
                            <p className="text-xs text-slate-400">Chart will be displayed here</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
