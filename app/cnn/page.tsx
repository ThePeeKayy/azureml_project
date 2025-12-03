"use client"

import type React from "react"

import { Navigation } from "@/components/navigation"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useState } from "react"
import {
  Play,
  AlertCircle,
  CheckCircle,
  Download,
  FileCode,
  FileJson,
  Clock,
  FolderOpen,
  X,
  Database,
  ImageIcon,
  Info,
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
  const [imagePreviews, setImagePreviews] = useState<string[]>([])

  // Dataset upload state
  const [datasetFile, setDatasetFile] = useState<File | null>(null)

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
  }

  const clearDataset = () => {
    setDatasetFile(null)
  }

  const loadSampleData = async () => {
    try {
      const response = await fetch("/pistachio.zip")
      if (!response.ok) throw new Error("Failed to load sample data")

      const blob = await response.blob()
      const file = new File([blob], "pistachio.zip", { type: "application/zip" })
      setDatasetFile(file)
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
    <TooltipProvider delayDuration={300}>
      <div className="min-h-screen bg-slate-950">
        <Navigation />
        <div className="pt-16 md:pt-24 pb-8 md:pb-12 px-3 md:px-4">
          <div className="container mx-auto max-w-7xl">
            <div className="mb-6 md:mb-8">
              <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-3 text-white">
                Custom Image Classification Training
              </h1>
              <p className="text-base md:text-lg text-slate-400 mb-4">
                Upload your image dataset and train a custom CNN model using MobileNetV3 or ShuffleNetV2 on Azure ML
              </p>

              <Card className="p-4 md:p-6 bg-transparent border-blue-800/30">
                <div className="flex items-start gap-3">
                  <div>
                    <p className="text-sm text-slate-300 leading-relaxed">
                      This playground page demonstrates end-to-end{" "}
                      <strong className="text-blue-300">deep learning workflow</strong> for custom image classification.
                      You upload a dataset of labeled images organized into class folders, select a lightweight CNN
                      architecture (MobileNetV3 or ShuffleNetV2), and configure hyperparameters like learning rate,
                      batch size, and epochs. When you start training, your dataset is uploaded to
                      <strong className="text-purple-300"> Azure Machine Learning</strong>, where a compute instance
                      trains the model using PyTorch. The training process includes data augmentation, 80/20
                      train-validation split, Adam optimization with learning rate scheduling, and real-time metrics
                      streaming. Once complete, the trained model weights, detailed metrics, and training history are
                      saved to
                      <strong className="text-cyan-300"> Azure Blob Storage</strong> and made available for
                      download.
                      <br />
                      <br />  
                      <span className="font-bold">CNN: </span>Convolutional Neural Networks use layers of learnable filters to automatically extract hierarchical spatial features from images through convolution operations, pooling, and nonlinear activations.
                      <br />
                      <span className="font-bold">MobileNet: </span>replaces standard convolutions with depthwise separable convolutions (splitting filtering and combining into two steps) to dramatically reduce parameters and computation while maintaining reasonable accuracy for mobile devices.
                      <br />
                      <span className="font-bold">ShuffleNet: </span>uses pointwise group convolutions combined with channel shuffle operations to enable information flow between channel groups, achieving even greater computational efficiency than MobileNet for extremely low-resource scenarios.
                    </p>
                  </div>
                </div>
              </Card>
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
                  <div className="flex items-center gap-2 mb-4 md:mb-6">
                    <h2 className="text-lg md:text-xl font-semibold text-white">Hyperparameters</h2>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-4 w-4 text-slate-500 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-xs">
                          Hyperparameters control how your model learns. Adjust these to optimize training speed and
                          accuracy.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-4 md:space-y-6">
                    {/* Model Type Selection */}
                    <div>
                      <div className="flex items-center gap-2 mb-2 md:mb-3">
                        <Label className="text-sm md:text-base text-slate-300">Model Architecture</Label>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-3 w-3 text-slate-500 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">
                              Choose between two efficient CNN architectures designed for mobile and edge deployment.
                              MobileNet uses depthwise separable convolutions, while ShuffleNet uses channel shuffle
                              operations.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
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
                        <div className="flex items-center gap-2">
                          <Label className="text-sm md:text-base text-slate-300">Learning Rate</Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-slate-500 cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="text-xs">
                                Controls how quickly the model adapts to the data. Higher values train faster but may
                                overshoot optimal weights. Lower values are more precise but slower. We use Adam
                                optimizer with automatic learning rate reduction on plateaus.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <span className="text-xs md:text-sm font-mono text-slate-400">
                          {learningRate[0].toFixed(4)}
                        </span>
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
                        <div className="flex items-center gap-2">
                          <Label className="text-sm md:text-base text-slate-300">Batch Size</Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-slate-500 cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="text-xs">
                                Number of images processed together in each training step. Larger batches provide more
                                stable gradient estimates and faster training but require more memory. Smaller batches
                                add noise that can help generalization.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
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
                        <div className="flex items-center gap-2">
                          <Label className="text-sm md:text-base text-slate-300">Training Epochs</Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-slate-500 cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="text-xs">
                                One epoch means the model has seen every image in your dataset once. More epochs allow
                                the model to learn better patterns but take longer and risk overfitting to training
                                data.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <span className="text-xs md:text-sm font-mono text-slate-400">{epochs[0]}</span>
                      </div>
                      <Slider
                        min={1}
                        max={10}
                        step={1}
                        value={epochs}
                        onValueChange={setEpochs}
                        disabled={isTraining}
                      />
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
                  <div className="flex items-center gap-2 mb-3 md:mb-4">
                    <h2 className="text-lg md:text-xl font-semibold text-white">Model Architecture</h2>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-4 w-4 text-slate-500 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-xs">
                          These lightweight CNNs are optimized for inference on mobile devices and edge hardware, using
                          techniques like depthwise convolutions to reduce parameters while maintaining accuracy.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
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
                  <div className="flex items-center gap-2 mb-3 md:mb-4">
                    <h2 className="text-lg md:text-xl font-semibold text-white">Upload Image Dataset</h2>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-4 w-4 text-slate-500 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-xs">
                          Upload a ZIP file with your images organized into folders by class. The folder names become
                          your class labels. Images are automatically resized to 224x224 and augmented with random crops
                          and flips during training.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>

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
                    
                    {/* Download Buttons */}
                    <div className="flex items-start gap-3 p-4 bg-slate-800 rounded-lg">
                      <CheckCircle className="h-5 w-5 md:h-6 md:w-6 text-blue-400 flex-shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <h3 className="text-base md:text-lg font-semibold text-blue-400 mb-2">Training Complete!</h3>

                        {/* Download Buttons */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 md:gap-3">
                          {trainingData.results.model_url && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  onClick={() => window.open(trainingData.results?.model_url, "_blank")}
                                  variant="outline"
                                  className="w-full bg-blue-950/30 border-blue-800/50 hover:text-white hover:bg-blue-900/30 text-blue-400 text-xs md:text-sm"
                                >
                                  <Download className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                                  Model (.pth)
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="text-xs">
                                  Download trained PyTorch model weights (.pth file) for deployment
                                </p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                          {trainingData.results.results_url && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  onClick={() => window.open(trainingData.results?.results_url, "_blank")}
                                  variant="outline"
                                  className="w-full bg-emerald-950/30 border-emerald-800/50 hover:text-white hover:bg-emerald-900/30 text-emerald-400 text-xs md:text-sm"
                                >
                                  <FileCode className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                                  Results
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="text-xs">
                                  Complete training results including hyperparameters and class names
                                </p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                          {trainingData.results.metrics_url && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  onClick={() => window.open(trainingData.results?.metrics_url, "_blank")}
                                  variant="outline"
                                  className="w-full bg-purple-950/30 border-purple-800/50 hover:text-white hover:bg-purple-900/30 text-purple-400 text-xs md:text-sm"
                                >
                                  <FileJson className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                                  Metrics
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="text-xs">Detailed training metrics and per-epoch history (JSON format)</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                          {trainingData.results.manifest_url && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  onClick={() => window.open(trainingData.results?.manifest_url, "_blank")}
                                  variant="outline"
                                  className="w-full bg-orange-950/30 border-orange-800/50 hover:text-white hover:bg-orange-900/30 text-orange-400 text-xs md:text-sm"
                                >
                                  <Database className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                                  Manifest
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="text-xs">Training manifest with job metadata and file URLs</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                        </div>
                      </div>
                    </div>
                  </Card>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}
