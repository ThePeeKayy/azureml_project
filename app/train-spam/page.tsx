"use client"

import { Navigation } from "@/components/navigation"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { useState, useEffect } from "react"
import { Play, Download, Shield, CheckCircle, AlertCircle, Send, Loader2, Database, Clock, FileCode, FileJson, BarChart3 } from "lucide-react"

interface TrainingUpdate {
  portalUrl?: string | URL
  jobId: string
  status: string
  progress?: number
  loss?: number
  accuracy?: number
  val_accuracy?: number
  currentEpoch?: number
  totalEpochs?: number
  timestamp: string
  results?: { 
    final_train_loss?: number | null
    final_train_accuracy?: number | null
    final_val_loss?: number | null
    final_val_accuracy?: number | null
    training_history?: { train_losses: number[], train_accuracies: number[], val_losses: number[], val_accuracies: number[] }
    model_url?: string
    results_url?: string
    metrics_url?: string
    manifest_url?: string
  }
  modelUrl?: string
  resultsUrl?: string
  metricsUrl?: string
  error?: string
  message?: string
}

export default function SpamDetectorPage() {
  const [hiddenUnits, setHiddenUnits] = useState([64])
  const [learningRate, setLearningRate] = useState([0.00002])
  const [epochs, setEpochs] = useState([3])
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
      const response = await fetch("/api/train-spam", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ hiddenUnits: hiddenUnits[0], learningRate: learningRate[0], epochs: epochs[0] }) })
      if (!response.ok) throw new Error("Failed to start training")
      if (!response.body) throw new Error("No response body")
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) { console.log("Stream ended"); break }
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
      error: { label: "Error", color: "bg-red-500/20 text-red-400 border-red-500/30" }
    }
    const statusInfo = statusMap[status] || statusMap.notstarted
    return <Badge variant="outline" className={`${statusInfo.color}`}>{statusInfo.label}</Badge>
  }

  return (
    <div className="min-h-screen bg-slate-950">
      <Navigation />
      <div className="pt-24 pb-12 px-4">
        <div className="container mx-auto max-w-7xl">
          <div className="mb-8">

            <h1 className="text-4xl font-bold mb-3 text-white">DistilBERT Spam Detector</h1>
            <p className="text-lg text-slate-400 max-w-3xl leading-relaxed">Train a DistilBERT transformer model on Azure ML using the UCI SMS Spam Collection dataset. Optimized for fast training with 1,000 sample subset.</p>
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
            <Card className="p-6 bg-slate-900 border-slate-800">
              <h2 className="text-xl font-semibold mb-6 text-white">Model Parameters</h2>
              <div className="space-y-6">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-slate-300">Hidden Units</Label>
                    <span className="text-sm font-mono text-slate-400">{hiddenUnits[0]}</span>
                  </div>
                  <Slider min={16} max={64} step={16} value={hiddenUnits} onValueChange={setHiddenUnits} disabled={isTraining} />
                  <p className="text-xs text-slate-500 mt-2">Classification layer neurons</p>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-slate-300">Learning Rate</Label>
                    <span className="text-sm font-mono text-slate-400">{learningRate[0].toFixed(5)}</span>
                  </div>
                  <Slider min={0.00001} max={0.0001} step={0.00001} value={learningRate} onValueChange={setLearningRate} disabled={isTraining} />
                  <p className="text-xs text-slate-500 mt-2">Adam optimizer (BERT requires lower LR)</p>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-slate-300">Training Epochs</Label>
                    <span className="text-sm font-mono text-slate-400">{epochs[0]}</span>
                  </div>
                  <Slider min={1} max={3} step={1} value={epochs} onValueChange={setEpochs} disabled={isTraining} />
                  <p className="text-xs text-slate-500 mt-2">BERT converges faster than simple NNs</p>
                </div>
                <div className="pt-4">
                  <Button onClick={handleTrain} disabled={isTraining} className="w-full bg-blue-600 hover:bg-blue-700">
                    {isTraining ? (
                      <>
                        <Shield className="h-4 w-4 mr-2 animate-pulse" />
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
              <div className="mt-6 pt-6 border-t border-slate-800">
                <h3 className="text-sm font-semibold mb-3 text-slate-300">Dataset & Configuration</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Source:</span>
                    <span className="font-mono text-xs text-slate-400">UCI ML Repository</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Dataset:</span>
                    <span className="font-mono text-xs text-slate-400">SMS Spam Collection</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Training Size:</span>
                    <span className="font-mono text-xs text-slate-400">1,000 messages (fast)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Model:</span>
                    <span className="font-mono text-xs text-slate-400">DistilBERT-base-uncased</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Framework:</span>
                    <span className="font-mono text-xs text-slate-400">PyTorch + Transformers</span>
                  </div>
                  {jobId && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">Job ID:</span>
                      <span className="font-mono text-xs text-blue-400 truncate max-w-[120px]" title={jobId}>{jobId.slice(-12)}</span>
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
            <div className="lg:col-span-2 space-y-6">
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
                  <p className="text-xs text-slate-500 mt-2">Training DistilBERT on 1,000 SMS messages with Azure ML CPU cluster</p>
                </Card>
              )}
              {trainingData?.status === "completed" && trainingData.results && (
                <Card className="p-6 bg-blue-950/50 border-gray-800 space-y-4">
                  <div className="flex items-start gap-3">
                    <CheckCircle className="h-6 w-6 text-blue-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-blue-400 mb-2">Training Complete!</h3>
                      
                      {trainingData.results?.training_history && trainingData.results.training_history.train_losses.length > 0 && (
                        <div className="bg-slate-900/50 rounded-lg p-3 space-y-2 mb-4">
                          <p className="text-xs font-semibold text-slate-400">Training History ({trainingData.results.training_history.train_losses.length} Epochs)</p>
                          <div className="space-y-1 text-xs">
                            <p><span className="text-slate-500">Train Losses:</span> <span className="font-mono text-emerald-300">{trainingData.results.training_history.train_losses.map(l => l.toFixed(4)).join(' → ')}</span></p>
                            <p><span className="text-slate-500">Train Accuracies:</span> <span className="font-mono text-purple-300">{trainingData.results.training_history.train_accuracies.map(a => `${(a * 100).toFixed(1)}%`).join(' → ')}</span></p>
                            <p><span className="text-slate-500">Val Losses:</span> <span className="font-mono text-blue-300">{trainingData.results.training_history.val_losses.map(l => l.toFixed(4)).join(' → ')}</span></p>
                            <p><span className="text-slate-500">Val Accuracies:</span> <span className="font-mono text-amber-300">{trainingData.results.training_history.val_accuracies.map(a => `${(a * 100).toFixed(1)}%`).join(' → ')}</span></p>
                          </div>
                        </div>
                      )}
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
                          <div className="text-white font-medium">Model Weights (.pt)</div>
                          <div className="text-xs text-slate-400">DistilBERT trained model checkpoint</div>
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
                          <div className="text-white font-medium">Training Results (.json)</div>
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
                          <div className="text-white font-medium">Metrics (.json)</div>
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
                          <div className="text-white font-medium">Manifest (.json)</div>
                          <div className="text-xs text-slate-400">Job metadata and output file registry</div>
                        </div>
                        <Download className="h-4 w-4 ml-2 text-slate-500" />
                      </Button>
                    )}
                  </div>
                </Card>
              )}
              
              <Card className="p-6 bg-slate-900 border-slate-800">
                <h2 className="text-xl font-semibold mb-4 text-white">DistilBERT Model Architecture</h2>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                    <span className="text-sm text-slate-300">DistilBERT Tokenizer</span>
                    <span className="text-xs font-mono text-slate-500">WordPiece (30K vocab)</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                    <span className="text-sm text-slate-300">DistilBERT Encoder</span>
                    <span className="text-xs font-mono text-slate-500">6 layers, 768 hidden</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                    <span className="text-sm text-slate-300">Pooler Output</span>
                    <span className="text-xs font-mono text-slate-500">[CLS] token (768 dims)</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                    <span className="text-sm text-slate-300">Dropout</span>
                    <span className="text-xs font-mono text-slate-500">p=0.3</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                    <span className="text-sm text-slate-300">Classification Layer</span>
                    <span className="text-xs font-mono text-slate-500">768 → {hiddenUnits[0]} → 2</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                    <span className="text-sm text-slate-300">Loss Function</span>
                    <span className="text-xs font-mono text-slate-500">Cross Entropy</span>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}