"use client"

import type React from "react"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useState, useEffect } from "react"
import {
  Play,
  Upload,
  AlertCircle,
  CheckCircle,
  FileSpreadsheet,
  X,
  Download,
  FileCode,
  FileJson,
  Database,
  Clock,
  Info,
} from "lucide-react"
import { Navigation } from "@/components/navigation"

export default function CustomTrainingPage() {
  const [hiddenUnits, setHiddenUnits] = useState([64])
  const [learningRate, setLearningRate] = useState([0.00002])
  const [epochs, setEpochs] = useState([3])
  const [isTraining, setIsTraining] = useState(false)
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [csvHeaders, setCsvHeaders] = useState<string[]>([])
  const [csvPreview, setCsvPreview] = useState<string[][]>([])
  const [dependentVariable, setDependentVariable] = useState("")
  const [error, setError] = useState<string | null>(null)

  const [trainingData, setTrainingData] = useState<any | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string | null>(null)

  const handleTrain = () => {
    setIsTraining(true)
    setError(null)
    setTrainingData(null) 
    setTimeout(() => {
      setCurrentStatus("running")
      setTimeout(() => {
        setCurrentStatus("waiting_for_outputs")
        setTimeout(() => {
          setIsTraining(false)
        }, 3000)
      }, 2000)
    }, 1000)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setCsvFile(file)
      setError(null)
      parseCsv(file)
    }
  }

  const parseCsv = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target?.result as string
      const lines = text.split("\n")
      const headers = lines[0].split(",").map((h) => h.trim())
      setCsvHeaders(headers)
      const preview = lines.slice(1, 6).map((line) => line.split(",").map((cell) => cell.trim()))
      setCsvPreview(preview)
      setDependentVariable("") 
    }
    reader.readAsText(file)
  }

  const loadSampleData = async () => {
    try {
      const response = await fetch("/spam.csv")
      if (!response.ok) throw new Error("Failed to load sample data")

      const text = await response.text()
      const lines = text.split("\n").filter((line) => line.trim())

      if (lines.length < 2) {
        setError("Sample CSV file is invalid")
        return
      }

      const headers = lines[0].split(",").map((h) => h.trim())
      setCsvHeaders(headers)

      const preview = lines.slice(1, 6).map((line) => line.split(",").map((cell) => cell.trim()))
      setCsvPreview(preview)

      const blob = new Blob([text], { type: "text/csv" })
      const file = new File([blob], "spam.csv", { type: "text/csv" })
      setCsvFile(file)

      setDependentVariable("")
      setError(null)
    } catch (err) {
      setError("Failed to load sample data. Make sure sample-data.csv is in the public folder.")
    }
  }

  const clearCsv = () => {
    setCsvFile(null)
    setCsvHeaders([])
    setCsvPreview([])
    setDependentVariable("")
    setError(null)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "running":
        return <Badge className="bg-blue-500/20 text-blue-400">Running</Badge>
      case "waiting_for_outputs":
        return <Badge className="bg-purple-500/20 text-purple-400">Waiting for Outputs</Badge>
      case "completed":
        return <Badge className="bg-emerald-500/20 text-emerald-400">Completed</Badge>
      case "failed":
        return <Badge className="bg-red-500/20 text-red-400">Failed</Badge>
      default:
        return <Badge className="bg-slate-700/20 text-slate-400">{status}</Badge>
    }
  }

  useEffect(() => {
    return () => {
      setIsTraining(false)
      setCurrentStatus(null)
      setTrainingData(null)
    }
  }, [])

  return (
    <TooltipProvider delayDuration={300}>
      <div className="min-h-screen bg-slate-950">
        <Navigation />
        <div className="pt-16 md:pt-24 pb-8 md:pb-12 px-3 md:px-4">
          <div className="container mx-auto max-w-7xl">
            <div className="mb-6 md:mb-8">
              <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-3 text-white">
                Custom Text Classification Training
              </h1>
              <p className="text-base md:text-lg text-slate-400 mb-4">
                Upload your CSV file and train a custom text classification model using DistilBERT on Azure ML
              </p>

              <Card className="p-4 md:p-6 bg-transparent border-blue-800/30">
                <div className="flex items-start gap-3">
                  <div>
                    <p className="text-sm text-slate-300 leading-relaxed">
                      This Azure AI playground demonstrates end-to-end{" "}
                      <strong className="text-blue-300">natural language processing (NLP) workflow</strong> for custom
                      text classification. You upload a CSV dataset containing text data and labels, select a dependent
                      variable (target column), and configure hyperparameters like hidden units, learning rate, and
                      epochs. When you start training, your dataset is uploaded to
                      <strong className="text-purple-300"> Azure Machine Learning</strong>, where a compute instance
                      fine-tunes a pre-trained DistilBERT model using PyTorch. The training process includes automatic
                      text tokenization with max sequence length of 64 tokens, 80/20 train-validation split with
                      stratification, AdamW optimization with linear warmup scheduling, gradient clipping for stability,
                      and real-time metrics streaming. Once complete, the fine-tuned model weights, detailed metrics,
                      and training history are saved to
                      <strong className="text-cyan-300"> Azure Blob Storage</strong> and made available for download.
                      <br />
                      <br />
                      <span className="font-bold">DistilBERT: </span>A distilled (smaller, faster) version of BERT that
                      retains 97% of BERT's language understanding while being 40% smaller and 60% faster. Uses 6
                      transformer layers (vs BERT's 12) with 66M parameters, making it ideal for production deployments
                      while maintaining strong performance on text classification tasks.
                      <br />
                      <span className="font-bold">Transfer Learning: </span>We start with DistilBERT pre-trained on
                      massive text corpora, then fine-tune it on your specific classification task. This allows the
                      model to leverage deep language understanding learned from billions of words while adapting to
                      your domain.
                      <br />
                      <span className="font-bold">Tokenization: </span>Text is converted into numerical tokens using
                      WordPiece tokenization with a 30,000 token vocabulary. Special tokens [CLS] and [SEP] mark
                      sentence boundaries, and the [CLS] token's final representation is used for classification.
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
                          accuracy for your specific text classification task.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-4 md:space-y-6">
                    <div>
                      <div className="flex items-center justify-between mb-2 md:mb-3">
                        <div className="flex items-center gap-2">
                          <Label className="text-sm md:text-base text-slate-300">Hidden Units</Label>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-slate-500 cursor-help" />
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="text-xs">
                                Number of neurons in the hidden layer between DistilBERT's 768-dimensional output and
                                the final 2-class prediction. More units can capture complex patterns but increase
                                training time and risk overfitting. Start with 64 for most tasks.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <span className="text-xs md:text-sm font-mono text-slate-400">{hiddenUnits[0]}</span>
                      </div>
                      <Slider
                        min={16}
                        max={128}
                        step={16}
                        value={hiddenUnits}
                        onValueChange={setHiddenUnits}
                        disabled={isTraining}
                      />
                      <p className="text-xs text-slate-500 mt-2">Number of neurons in the hidden layer</p>
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
                                Controls how quickly the model adapts to your data during fine-tuning. For DistilBERT,
                                we use very small learning rates (2e-5 recommended) to avoid catastrophic forgetting of
                                pre-trained knowledge. We use AdamW optimizer with linear warmup and decay for stable
                                training.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <span className="text-xs md:text-sm font-mono text-slate-400">
                          {learningRate[0].toFixed(5)}
                        </span>
                      </div>
                      <Slider
                        min={0.00001}
                        max={0.001}
                        step={0.00001}
                        value={learningRate}
                        onValueChange={setLearningRate}
                        disabled={isTraining}
                      />
                      <p className="text-xs text-slate-500 mt-2">AdamW optimizer with linear warmup scheduler</p>
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
                                One epoch means the model has seen every text sample in your dataset once. For
                                fine-tuning pre-trained transformers like DistilBERT, 2-3 epochs are usually sufficient
                                since the model already understands language. More epochs risk overfitting.
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
                      disabled={isTraining || !csvFile || !dependentVariable}
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
                    <h2 className="text-lg md:text-xl font-semibold text-white">DistilBERT Architecture</h2>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-4 w-4 text-slate-500 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-xs">
                          DistilBERT is a streamlined transformer model that uses self-attention mechanisms to process
                          text bidirectionally. It's been distilled from BERT through knowledge distillation during
                          pre-training, maintaining high performance with significantly reduced computational
                          requirements.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <div className="space-y-2 md:space-y-3">
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">DistilBERT Encoder</span>
                      <span className="text-[10px] md:text-xs font-mono text-slate-500">6 layers, 768 hidden</span>
                    </div>
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">Pooler Output</span>
                      <span className="text-[10px] md:text-xs font-mono text-slate-500">[CLS] token (768 dims)</span>
                    </div>
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">Dropout</span>
                      <span className="text-[10px] md:text-xs font-mono text-slate-500">p=0.3</span>
                    </div>
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">Classification Layer</span>
                      <span className="text-[10px] md:text-xs font-mono text-slate-500">
                        768 → {hiddenUnits[0]} → 2
                      </span>
                    </div>
                    <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                      <span className="text-xs md:text-sm text-slate-300">Loss Function</span>
                      <span className="text-[10px] md:text-xs font-mono text-slate-500">Cross Entropy</span>
                    </div>
                  </div>
                  <div className="mt-3 md:mt-4 p-2 md:p-3 bg-blue-950/30 border border-blue-800/30 rounded-lg">
                    <p className="text-xs text-blue-300">
                      <strong>Optimizer:</strong> AdamW with gradient clipping (max_norm=1.0) and linear warmup
                      scheduler
                    </p>
                  </div>
                </Card>
              </div>
              <div className="lg:col-span-2 space-y-4 md:space-y-6">
                <Card className="p-4 md:p-6 bg-slate-900 border-slate-800">
                  <div className="flex items-center gap-2 mb-3 md:mb-4">
                    <h2 className="text-lg md:text-xl font-semibold text-white">Upload Dataset</h2>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Info className="h-4 w-4 text-slate-500 cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-xs">
                          Upload a CSV file with text data and labels. All columns except the dependent variable will be
                          concatenated as input text. The model automatically tokenizes text to max 64 tokens, applies
                          stratified train-validation split (80/20), and handles class imbalance through proper
                          sampling.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>

                  {!csvFile ? (
                    <div>
                      <div className="border-2 border-dashed border-slate-700 rounded-lg p-6 md:p-8 text-center hover:border-slate-600 transition-colors">
                        <input
                          type="file"
                          accept=".csv"
                          onChange={handleFileUpload}
                          className="hidden"
                          id="csv-upload"
                        />
                        <label htmlFor="csv-upload" className="cursor-pointer">
                          <Upload className="h-10 w-10 md:h-12 md:w-12 text-slate-500 mx-auto mb-3 md:mb-4" />
                          <p className="text-sm md:text-base text-slate-300 mb-1 md:mb-2">Click to upload CSV file</p>
                          <p className="text-xs md:text-sm text-slate-500">CSV files only</p>
                        </label>
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
                        <FileSpreadsheet className="h-4 w-4 mr-2 text-white" />
                        Load Sample Dataset (Spam dataset from Kaggle)
                      </Button>
                    </div>
                  ) : (
                    <div>
                      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-3 md:p-4 bg-slate-800 rounded-lg mb-3 md:mb-4 gap-2">
                        <div className="flex items-center gap-2 md:gap-3">
                          <FileSpreadsheet className="h-4 w-4 md:h-5 md:w-5 text-blue-400 flex-shrink-0" />
                          <div>
                            <p className="text-xs md:text-sm font-medium text-white break-all">{csvFile.name}</p>
                            <p className="text-[10px] md:text-xs text-slate-500">
                              {csvHeaders.length} columns, {csvPreview.length} rows preview
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 w-full sm:w-auto">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={clearCsv}
                            disabled={isTraining}
                            className="text-slate-400 hover:text-red-400 text-xs md:text-sm flex-1 sm:flex-none"
                          >
                            <X className="h-3 w-3 md:h-4 md:w-4" />
                            <span className="ml-1">Remove</span>
                          </Button>
                        </div>
                      </div>

                      <div className="mb-3 md:mb-4">
                        <Label className="text-sm md:text-base text-slate-300 mb-2 block">
                          Select Dependent Variable
                        </Label>
                        <Select value={dependentVariable} onValueChange={setDependentVariable} disabled={isTraining}>
                          <SelectTrigger className="bg-slate-800 border-slate-700 text-white text-sm md:text-base">
                            <SelectValue placeholder="Choose the target column" />
                          </SelectTrigger>
                          <SelectContent>
                            {csvHeaders.map((header) => (
                              <SelectItem key={header} value={header}>
                                {header}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-slate-500 mt-2">
                          This is the column you want to predict (must have exactly 2 unique values for binary
                          classification)
                        </p>
                      </div>

                      <div>
                        <Label className="text-sm md:text-base text-slate-300 mb-2 block">
                          Data Preview (First 5 Rows)
                        </Label>
                        <div className="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0">
                          <div className="min-w-[600px]">
                            <table className="w-full text-xs md:text-sm border border-slate-700 rounded-lg overflow-hidden">
                              <thead className="bg-slate-800">
                                <tr>
                                  {csvHeaders.map((header, idx) => (
                                    <th
                                      key={idx}
                                      className={`px-2 md:px-4 py-2 text-left font-medium whitespace-nowrap ${
                                        header === dependentVariable ? "text-blue-400 bg-blue-950/30" : "text-slate-300"
                                      }`}
                                    >
                                      {header}
                                      {header === dependentVariable && (
                                        <Badge className="ml-1 md:ml-2 text-[10px] md:text-xs bg-blue-600">
                                          Target
                                        </Badge>
                                      )}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {csvPreview.map((row, rowIdx) => (
                                  <tr key={rowIdx} className="border-t border-slate-800">
                                    {row.map((cell, cellIdx) => (
                                      <td
                                        key={cellIdx}
                                        className={`px-2 md:px-4 py-2 ${
                                          csvHeaders[cellIdx] === dependentVariable
                                            ? "bg-blue-950/20 text-blue-300"
                                            : "text-slate-400"
                                        }`}
                                      >
                                        {cell}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
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
                    <p className="text-xs text-slate-500 mt-2">
                      Fine-tuning DistilBERT on your text classification task via Azure ML
                    </p>
                  </Card>
                )}

                {trainingData?.status === "completed" && trainingData.results && (
                  <Card className="p-4 md:p-6 bg-gray-900 border-gray-800 space-y-3 md:space-y-4">
                    <div className="flex items-start gap-3 p-4 bg-slate-800 rounded-lg">
                      <CheckCircle className="h-5 w-5 md:h-6 md:w-6 text-blue-400 flex-shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <h3 className="text-base md:text-lg font-semibold text-blue-400 mb-2">Training Complete!</h3>

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
                                  Model (.pt)
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="text-xs">
                                  Download fine-tuned DistilBERT model weights (.pt file) for deployment and inference
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
                                  Complete training results including hyperparameters, label mappings, and class
                                  distributions
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
