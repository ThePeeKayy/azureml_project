"use client"

import type React from "react"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useState } from "react"
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
} from "lucide-react"
import { Navigation } from "@/components/navigation"

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

export default function CustomTrainingPage() {
  // Model parameters
  const [hiddenUnits, setHiddenUnits] = useState([64])
  const [learningRate, setLearningRate] = useState([0.00002])
  const [epochs, setEpochs] = useState([3])

  // CSV data
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [csvHeaders, setCsvHeaders] = useState<string[]>([])
  const [csvPreview, setCsvPreview] = useState<string[][]>([])
  const [dependentVariable, setDependentVariable] = useState<string>("")

  // Training state
  const [isTraining, setIsTraining] = useState(false)
  const [trainingData, setTrainingData] = useState<TrainingUpdate | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string>("")

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setCsvFile(file)
    setError(null)

    const reader = new FileReader()
    reader.onload = (event) => {
      const text = event.target?.result as string
      const lines = text.split("\n").filter((line) => line.trim())

      if (lines.length < 2) {
        setError("CSV file must have at least a header row and one data row")
        return
      }

      // Parse headers
      const headers = lines[0].split(",").map((h) => h.trim())
      setCsvHeaders(headers)

      // Parse first 5 rows for preview
      const preview = lines.slice(1, 6).map((line) => line.split(",").map((cell) => cell.trim()))
      setCsvPreview(preview)

      // Reset dependent variable selection
      setDependentVariable("")
    }

    reader.readAsText(file)
  }

  const clearCsv = () => {
    setCsvFile(null)
    setCsvHeaders([])
    setCsvPreview([])
    setDependentVariable("")
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

      // Parse headers
      const headers = lines[0].split(",").map((h) => h.trim())
      setCsvHeaders(headers)

      // Parse first 5 rows for preview
      const preview = lines.slice(1, 6).map((line) => line.split(",").map((cell) => cell.trim()))
      setCsvPreview(preview)

      // Create a File object from the text
      const blob = new Blob([text], { type: "text/csv" })
      const file = new File([blob], "sample-data.csv", { type: "text/csv" })
      setCsvFile(file)

      // Reset dependent variable
      setDependentVariable("")
      setError(null)
    } catch (err) {
      setError("Failed to load sample data. Make sure sample-data.csv is in the public folder.")
    }
  }

  const handleTrain = async () => {
    if (!csvFile || !dependentVariable) {
      setError("Please upload a CSV file and select a dependent variable")
      return
    }

    setIsTraining(true)
    setError(null)
    setTrainingData(null)
    setCurrentStatus("")

    try {
      const formData = new FormData()
      formData.append("file", csvFile)
      formData.append("dependentVariable", dependentVariable)
      formData.append("hiddenUnits", hiddenUnits[0].toString())
      formData.append("learningRate", learningRate[0].toString())
      formData.append("epochs", epochs[0].toString())

      const response = await fetch("/api/train-custom", {
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
            <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-3 text-white">Custom NLP DistilBert training</h1>
            <p className="text-base md:text-lg text-slate-400">
              Upload your CSV file and train a custom classification model using DistilBERT on Azure ML
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
            <div className="space-y-4 md:space-y-6">
              <Card className="p-4 md:p-6 bg-slate-900 border-slate-800">
                <h2 className="text-lg md:text-xl font-semibold mb-4 md:mb-6 text-white">Hyperparameters</h2>
                <div className="space-y-4 md:space-y-6">
                  <div>
                    <div className="flex items-center justify-between mb-2 md:mb-3">
                      <Label className="text-sm md:text-base text-slate-300">Hidden Units</Label>
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
                      <Label className="text-sm md:text-base text-slate-300">Learning Rate</Label>
                      <span className="text-xs md:text-sm font-mono text-slate-400">{learningRate[0].toFixed(5)}</span>
                    </div>
                    <Slider
                      min={0.00001}
                      max={0.001}
                      step={0.00001}
                      value={learningRate}
                      onValueChange={setLearningRate}
                      disabled={isTraining}
                    />
                    <p className="text-xs text-slate-500 mt-2">Step size for gradient descent optimization</p>
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
                <h2 className="text-lg md:text-xl font-semibold mb-3 md:mb-4 text-white">DistilBERT Architecture</h2>
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
                    <span className="text-[10px] md:text-xs font-mono text-slate-500">768 → {hiddenUnits[0]} → 2</span>
                  </div>
                  <div className="flex items-center justify-between p-2 md:p-3 bg-slate-800 rounded-lg">
                    <span className="text-xs md:text-sm text-slate-300">Loss Function</span>
                    <span className="text-[10px] md:text-xs font-mono text-slate-500">Cross Entropy</span>
                  </div>
                </div>
              </Card>
            </div>
            <div className="lg:col-span-2 space-y-4 md:space-y-6">
              <Card className="p-4 md:p-6 bg-slate-900 border-slate-800">
                <h2 className="text-lg md:text-xl font-semibold mb-3 md:mb-4 text-white">Upload Dataset</h2>

                {!csvFile ? (
                  <div>
                    <div className="border-2 border-dashed border-slate-700 rounded-lg p-6 md:p-8 text-center hover:border-slate-600 transition-colors">
                      <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" id="csv-upload" />
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
                      <p className="text-xs text-slate-500 mt-2">This is the column you want to predict</p>
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
                                      <Badge className="ml-1 md:ml-2 text-[10px] md:text-xs bg-blue-600">Target</Badge>
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
                  <p className="text-xs text-slate-500 mt-2">Training your custom NLP classifier on Azure ML</p>
                </Card>
              )}

              {trainingData?.status === "completed" && trainingData.results && (
                <Card className="p-4 md:p-6 bg-gray-900 border-gray-800 space-y-3 md:space-y-4">
                  <div className="flex items-start gap-2 md:gap-3">
                    <CheckCircle className="h-5 w-5 md:h-6 md:w-6 text-blue-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <h3 className="text-base md:text-lg font-semibold text-blue-400 mb-2">Training Complete!</h3>

                      {/* Download Buttons */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 md:gap-3">
                        {trainingData.results.model_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.model_url, "_blank")}
                            variant="outline"
                            className="w-full bg-blue-950/30 border-blue-800/50 hover:text-white hover:bg-blue-900/30 text-blue-400 text-xs md:text-sm"
                          >
                            <Download className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Model (.pt)
                          </Button>
                        )}
                        {trainingData.results.results_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.results_url, "_blank")}
                            variant="outline"
                            className="w-full bg-emerald-950/30 border-emerald-800/50 hover:text-white hover:bg-emerald-900/30 text-emerald-400 text-xs md:text-sm"
                          >
                            <FileCode className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Results
                          </Button>
                        )}
                        {trainingData.results.metrics_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.metrics_url, "_blank")}
                            variant="outline"
                            className="w-full bg-purple-950/30 border-purple-800/50 hover:text-white hover:bg-purple-900/30 text-purple-400 text-xs md:text-sm"
                          >
                            <FileJson className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Metrics
                          </Button>
                        )}
                        {trainingData.results.manifest_url && (
                          <Button
                            onClick={() => window.open(trainingData.results?.manifest_url, "_blank")}
                            variant="outline"
                            className="w-full bg-orange-950/30 border-orange-800/50 hover:text-white hover:bg-orange-900/30 text-orange-400 text-xs md:text-sm"
                          >
                            <Database className="h-3 w-3 md:h-4 md:w-4 mr-1 md:mr-2" />
                            Manifest
                          </Button>
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
  )
}
