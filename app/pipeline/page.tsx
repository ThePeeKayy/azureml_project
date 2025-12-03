"use client"

import type React from "react"
import { Navigation } from "@/components/navigation"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useState, useEffect } from "react"
import Image from 'next/image'
import { 
  Download, Plus, GripVertical, Edit2, Trash2, Copy, Upload, 
  Database, Workflow, Play, Clock, Zap, RefreshCw, CheckCircle, 
  XCircle, Loader2, Table, AlertCircle, Plane, Activity, MapPin,
  TrendingUp, Navigation as NavigationIcon, Gauge
} from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"

type PipelineActivity = {
  id: string
  type: string
  name: string
  activityType: string
  description: string
}

type TriggerConfig = {
  id: string
  name: string
  type: "schedule" | "tumbling-window" | "storage-event"
  status: "Active" | "Inactive"
  lastRun?: string
  nextRun?: string
}

type PipelineRun = {
  runId: string
  status: "Succeeded" | "Failed" | "InProgress" | "Queued"
  startTime: string
  endTime?: string
  duration?: number
}

type Flight = {
  icao24: string
  callsign: string | null
  origin_country: string
  longitude: number | null
  latitude: number | null
  baro_altitude: number | null
  velocity: number | null
  vertical_rate: number | null
  on_ground: boolean
}

type FlightData = {
  latest_update: string
  file_name: string
  file_size_kb: number
  status: string
  flight_count?: number
  flights?: Flight[]
}

export default function ADFPipelinePage() {
  // State for pipeline design
  const [activities, setActivities] = useState<PipelineActivity[]>([
    {
      id: "1",
      type: "copy",
      name: "Copy Flight Data",
      activityType: "Copy",
      description: "Ingest real-time flight data from OpenSky Network API to Blob Storage"
    },
    {
      id: "2",
      type: "dataflow",
      name: "Transform Flight Data",
      activityType: "Data Flow",
      description: "Process and enrich flight geography data (lat, long, altitude, velocity)"
    },
    {
      id: "3",
      type: "delete",
      name: "Archive Old Files",
      activityType: "Delete",
      description: "Clean up raw files older than 7 days"
    }
  ])

  const [triggers, setTriggers] = useState<TriggerConfig[]>([])
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRun[]>([])
  const [flightData, setFlightData] = useState<FlightData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isTriggering, setIsTriggering] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)

  // API endpoint configuration
  const API_BASE_URL = process.env.NEXT_PUBLIC_ADF_API_URL || "http://localhost:7071/api"

  const [draggedActivity, setDraggedActivity] = useState<string | null>(null)
  const [editingActivity, setEditingActivity] = useState<PipelineActivity | null>(null)

  const activityTypes = [
    { 
      value: "copy", 
      label: "Copy Data", 
      color: "bg-blue-500", 
      icon: "📥",
      activityType: "Copy",
      description: "Copy data between data stores"
    },
    { 
      value: "dataflow", 
      label: "Data Flow", 
      color: "bg-emerald-500", 
      icon: "🔄",
      activityType: "Data Flow",
      description: "Transform data with mapping flows"
    },
    { 
      value: "databricks", 
      label: "Databricks Notebook", 
      color: "bg-orange-500", 
      icon: "📓",
      activityType: "Databricks Notebook",
      description: "Execute Spark notebooks"
    },
    { 
      value: "delete", 
      label: "Delete", 
      color: "bg-red-500", 
      icon: "🗑️",
      activityType: "Delete",
      description: "Delete files from storage"
    },
    { 
      value: "lookup", 
      label: "Lookup", 
      color: "bg-purple-500", 
      icon: "🔍",
      activityType: "Lookup",
      description: "Query data for pipeline logic"
    },
    { 
      value: "execute-pipeline", 
      label: "Execute Pipeline", 
      color: "bg-cyan-500", 
      icon: "⚡",
      activityType: "Execute Pipeline",
      description: "Invoke another pipeline"
    }
  ]

  const fetchTriggers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/triggers`)
      if (!response.ok) throw new Error("Failed to fetch triggers")
      const data = await response.json()
      setTriggers(data.triggers || [])
    } catch (err) {
      console.error("Error fetching triggers:", err)
      setTriggers([])
    }
  }

  // Fetch flight data
  const fetchFlightData = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/flights`)
      if (!response.ok) throw new Error("Failed to fetch flight data")
      const data = await response.json()
      setFlightData(data)
      setLastRefresh(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch data")
      // Fallback mock data with sample flights
      setFlightData({
        latest_update: new Date().toISOString(),
        file_name: "flights_demo.parquet",
        file_size_kb: 245.6,
        status: "success",
        flight_count: 5,
        flights: []
      })
      setLastRefresh(new Date())
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch pipeline runs
  const fetchPipelineRuns = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/pipeline-status`)
      if (!response.ok) throw new Error("Failed to fetch pipeline status")
      const data = await response.json()
      setPipelineRuns(data.runs || [])
    } catch (err) {
      console.error("Error fetching pipeline runs:", err)
      // Fallback mock data
      setPipelineRuns([
        {
          runId: `run-${new Date().getTime()}`,
          status: "Succeeded",
          startTime: new Date(Date.now() - 600000).toISOString(),
          endTime: new Date(Date.now() - 480000).toISOString(),
          duration: 120
        }
      ])
    }
  }

  // Trigger pipeline manually
  const triggerPipeline = async () => {
    setIsTriggering(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/trigger-pipeline`, {
        method: "POST"
      })
      if (!response.ok) throw new Error("Failed to trigger pipeline")
      const result = await response.json()
      
      // Refresh data after triggering
      setTimeout(() => {
        fetchFlightData()
        fetchPipelineRuns()
      }, 2000)
      
      alert(`Pipeline triggered! Run ID: ${result.runId}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to trigger pipeline")
      alert("Pipeline trigger simulated (API not connected)")
    } finally {
      setIsTriggering(false)
    }
  }

  // Load data on mount
  useEffect(() => {
    fetchTriggers()
    fetchFlightData()
    fetchPipelineRuns()
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      fetchTriggers()
      fetchFlightData()
      fetchPipelineRuns()
    }, 30000)
    
    return () => clearInterval(interval)
  }, [])

  const addActivity = (type: string) => {
    const activityType = activityTypes.find((a) => a.value === type)
    const newActivity: PipelineActivity = {
      id: Date.now().toString(),
      type,
      name: activityType?.label || "New Activity",
      activityType: activityType?.activityType || "Activity",
      description: activityType?.description || ""
    }
    setActivities([...activities, newActivity])
  }

  const removeActivity = (id: string) => {
    setActivities(activities.filter((a) => a.id !== id))
  }

  const duplicateActivity = (activity: PipelineActivity) => {
    const newActivity = { 
      ...activity, 
      id: Date.now().toString(), 
      name: `${activity.name} (Copy)` 
    }
    setActivities([...activities, newActivity])
  }

  const updateActivity = (updatedActivity: PipelineActivity) => {
    setActivities(activities.map((a) => (a.id === updatedActivity.id ? updatedActivity : a)))
    setEditingActivity(null)
  }

  const handleDragStart = (id: string) => {
    setDraggedActivity(id)
  }

  const handleDragOver = (e: React.DragEvent, targetId: string) => {
    e.preventDefault()
    if (!draggedActivity || draggedActivity === targetId) return

    const draggedIndex = activities.findIndex((a) => a.id === draggedActivity)
    const targetIndex = activities.findIndex((a) => a.id === targetId)

    const newActivities = [...activities]
    const [removed] = newActivities.splice(draggedIndex, 1)
    newActivities.splice(targetIndex, 0, removed)

    setActivities(newActivities)
  }

  const handleDragEnd = () => {
    setDraggedActivity(null)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "Succeeded": return <CheckCircle className="h-4 w-4 text-emerald-400" />
      case "Failed": return <XCircle className="h-4 w-4 text-red-400" />
      case "InProgress": return <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
      case "Queued": return <Clock className="h-4 w-4 text-yellow-400" />
      default: return <AlertCircle className="h-4 w-4 text-slate-400" />
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString("en-SG", {
      timeZone: "Asia/Singapore",
      dateStyle: "medium",
      timeStyle: "short"
    })
  }

  const formatAltitude = (altitude: number | null) => {
    if (altitude === null) return "N/A"
    return `${(altitude * 3.28084).toFixed(0)} ft`
  }

  const formatVelocity = (velocity: number | null) => {
    if (velocity === null) return "N/A"
    return `${(velocity * 1.94384).toFixed(0)} knots`
  }

  const formatCoordinates = (lat: number | null, lon: number | null) => {
    if (lat === null || lon === null) return "N/A"
    return `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`
  }

  return (
    <div className="min-h-screen bg-slate-950 pb-[100px] md:pb-0">
      <Navigation />

      <div className="pt-24 pb-12 px-4">
        <div className="container mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-8">
            
            <h1 className="text-4xl font-bold mb-3 text-white">Azure Data Factory - Live Pipeline</h1>
            <p className="text-lg text-slate-400 max-w-3xl leading-relaxed">
              Real-time flight tracking data pipeline with OpenSky Network API. Features 3 trigger types (Schedule, 
              Tumbling Window, Storage Event), streaming flight data, and live monitoring.
            </p>
          </div>

<Card className="p-4 md:p-6 bg-transparent border-blue-800/30">
  <div className="flex items-start gap-3">
    <div>
      <p className="text-sm text-slate-300 leading-relaxed">
        This playground page demonstrates end-to-end{" "}
        <strong className="text-blue-300">data orchestration</strong> for real-time flight tracking.
        The pipeline ingests live flight data from the OpenSky Network API, which provides aircraft positions,
        velocities, altitudes, and metadata for thousands of active flights worldwide. Using
        <strong className="text-purple-300"> Azure Data Factory</strong>, the pipeline executes a sequence of
        activities: a Copy activity fetches JSON data from the API, a Data Flow activity transforms and enriches
        the geography data (latitude, longitude, altitude, velocity), and a Delete activity archives old raw files.
        The processed data is stored as Parquet files with Snappy compression in
        <strong className="text-cyan-300"> Azure Blob Storage</strong>. The pipeline supports three trigger types—
        Schedule (fixed intervals), Tumbling Window (non-overlapping time windows), and Storage Event (blob
        creation/modification)—allowing flexible automation. You can manually trigger runs, monitor execution status
        in real-time, and view live flight details including callsigns, positions, altitudes, and ground status.
        <br />
        <br />  
        <span className="font-bold">Azure Data Factory: </span>A cloud-based ETL (Extract, Transform, Load) service that orchestrates data movement and transformation across hybrid data estates using visual pipelines with built-in activities, triggers, and monitoring.
        <br />
        <span className="font-bold">Schedule Trigger: </span>Executes pipelines at fixed time intervals (e.g., every 15 minutes) using cron-like expressions, ideal for regular batch processing and periodic data ingestion.
        <br />
        <span className="font-bold">Tumbling Window Trigger: </span>Creates non-overlapping, fixed-size time windows that process data in sequential chunks, ensuring exactly-once execution per window and enabling backfill for historical data processing.
        <br />
        <span className="font-bold">Storage Event Trigger: </span>Responds to blob storage events (create, delete) in real-time, enabling event-driven architectures where pipelines automatically process new files as they arrive in storage containers.
      </p>
    </div>
  </div>
</Card>
          {/* Error Alert */}
          {error && (
            <Alert className="mb-6 border-yellow-500/30 bg-yellow-500/10">
              <AlertCircle className="h-4 w-4 text-yellow-400" />
              <AlertDescription className="text-yellow-400">
                {error} - Using demo mode
              </AlertDescription>
            </Alert>
          )}

          <div className="grid lg:grid-cols-4 gap-3">
            {/* Left Sidebar */}
            <Card className="p-4 bg-slate-900 border-slate-800">
              <h2 className="text-xl font-semibold text-white">Controls</h2>
              
              {/* Active Triggers */}
              <div className="pt-6 border-t border-slate-800">
                <h3 className="text-sm font-semibold mb-3 text-slate-300">Live Triggers</h3>
                <div className="space-y-2">
                  {triggers.map((trigger) => (
                    <div
                      key={trigger.id}
                      className="p-3 rounded-lg border bg-slate-800/50 border-slate-700"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <Badge 
                          variant={"default"}
                          className="text-xs"
                        >
                          {trigger.status}
                        </Badge>
                      </div>
                      <p className="text-xs text-slate-400 mb-1">{trigger.name}</p>

                    </div>
                  ))}
                </div>
              </div>

              {/* Activity Types */}
              <h3 className="text-sm font-semibold mb-3 text-slate-300">Add activities to builder</h3>
              <div className="space-y-2 mb-6">
                {activityTypes.map((type) => (
                  <Button
                    key={type.value}
                    onClick={() => addActivity(type.value)}
                    variant="outline"
                    className="w-full justify-start border-slate-700 hover:bg-slate-800 hover:text-white text-left"
                    size="sm"
                  >
                    <div className={`w-2 h-2 rounded-full ${type.color} mr-2 flex-shrink-0`} />
                    <span className="text-xs">{type.icon} {type.label}</span>
                  </Button>
                ))}
              </div>

              {/* Info */}
              <div className="mt-6 p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                <div className="text-xs text-slate-400 space-y-1">
                  <div className="flex justify-between">
                    <span>Platform:</span>
                    <span className="text-slate-300">Azure ADF</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Storage:</span>
                    <span className="text-slate-300">Blob (5GB free)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Data Source:</span>
                    <span className="text-slate-300">OpenSky API</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cost:</span>
                    <span className="text-emerald-300 font-semibold">$0/month</span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Main Content */}
            <div className="lg:col-span-3 space-y-6">
              {/* Pipeline Design */}

              <Card className="p-6 bg-slate-900 border-slate-800">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-white">Live Pipeline Output</h2>
                    <p className="text-sm text-slate-400 mt-1">Real-time flight tracking from OpenSky Network (display only 50 to prevent lag)</p>
                  </div>
                  <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Live
                  </Badge>
                </div>

                {/* Quick Actions */}
              <div className="space-y-2 mb-6">
                <Button
                  onClick={triggerPipeline}
                  disabled={isTriggering}
                  className="w-full bg-blue-600 hover:bg-blue-700"
                >
                  {isTriggering ? (
                    <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Triggering...</>
                  ) : (
                    <><Play className="h-4 w-4 mr-2" /> Run Pipeline Now (default)</>
                  )}
                </Button>
                <Button
                  onClick={fetchFlightData}
                  disabled={isLoading}
                  variant="outline"
                  className="w-full border-slate-700"
                >
                  {isLoading ? (
                    <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Loading...</>
                  ) : (
                    <><RefreshCw className="h-4 w-4 mr-2" /> Refresh Data</>
                  )}
                </Button>
              </div>

                <Image
                  src="/DFpipeline.png"
                  width={1000}
                  height={1000}
                  alt="Picture of the author"
                  className="rounded-md mb-6"
                />

                <Tabs defaultValue="flights" className="w-full">
                  <TabsList className="grid w-full grid-cols-3 bg-slate-800">
                    <TabsTrigger value="flights" className="text-black data-[state=inactive]:text-white/70">
                      Live Flights
                    </TabsTrigger>
                    <TabsTrigger value="summary" className="text-black data-[state=inactive]:text-white/70">
                      Summary
                    </TabsTrigger>
                    <TabsTrigger value="runs" className="text-black data-[state=inactive]:text-white/70">
                      Pipeline Runs
                    </TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="flights" className="mt-4">
                    {flightData && flightData.flights && flightData.flights.length > 0 ? (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between mb-3">
                          <div className="text-sm text-slate-400">
                            Showing {flightData.flights.length} active flights
                          </div>
                        </div>
                        <ScrollArea className="h-[500px] pr-4">
                          <div className="space-y-3">
                            {flightData.flights.map((flight, index) => (
                              <Card key={index} className="p-4 bg-slate-800/50 border-slate-700 hover:border-slate-600 transition-colors">
                                <div className="flex items-start justify-between mb-3">
                                  <div className="flex items-center gap-3">
                                    <div className={`p-2 rounded-lg ${flight.on_ground ? 'bg-slate-700' : 'bg-blue-500/20'}`}>
                                      <Plane className={`h-5 w-5 ${flight.on_ground ? 'text-slate-400' : 'text-blue-400'}`} />
                                    </div>
                                    <div>
                                      <div className="flex items-center gap-2">
                                        <span className="font-semibold text-white text-lg">
                                          {flight.callsign?.trim() || "N/A"}
                                        </span>
                                        <Badge 
                                          variant="outline" 
                                          className={flight.on_ground ? "border-slate-600 text-slate-400" : "border-emerald-500/30 text-emerald-400"}
                                        >
                                          {flight.on_ground ? "On Ground" : "In Flight"}
                                        </Badge>
                                      </div>
                                      <div className="text-xs text-slate-400 mt-1 font-mono">
                                        ICAO24: {flight.icao24}
                                      </div>
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-sm text-slate-300 font-semibold">{flight.origin_country}</div>
                                  </div>
                                </div>
                                
                                <div className="grid grid-cols-2 gap-3 mt-3">
                                  <div className="flex items-center gap-2">
                                    <MapPin className="h-4 w-4 text-blue-400 flex-shrink-0" />
                                    <div>
                                      <div className="text-xs text-slate-400">Position</div>
                                      <div className="text-sm text-slate-300 font-mono">
                                        {formatCoordinates(flight.latitude, flight.longitude)}
                                      </div>
                                    </div>
                                  </div>
                                  
                                  <div className="flex items-center gap-2">
                                    <TrendingUp className="h-4 w-4 text-emerald-400 flex-shrink-0" />
                                    <div>
                                      <div className="text-xs text-slate-400">Altitude</div>
                                      <div className="text-sm text-slate-300 font-mono">
                                        {formatAltitude(flight.baro_altitude)}
                                      </div>
                                    </div>
                                  </div>
                                  
                                  <div className="flex items-center gap-2">
                                    <Gauge className="h-4 w-4 text-purple-400 flex-shrink-0" />
                                    <div>
                                      <div className="text-xs text-slate-400">Velocity</div>
                                      <div className="text-sm text-slate-300 font-mono">
                                        {formatVelocity(flight.velocity)}
                                      </div>
                                    </div>
                                  </div>
                                  
                                  <div className="flex items-center gap-2">
                                    <NavigationIcon className="h-4 w-4 text-yellow-400 flex-shrink-0" />
                                    <div>
                                      <div className="text-xs text-slate-400">Vertical Rate</div>
                                      <div className="text-sm text-slate-300 font-mono">
                                        {flight.vertical_rate !== null ? `${flight.vertical_rate.toFixed(1)} m/s` : "N/A"}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </Card>
                            ))}
                          </div>
                        </ScrollArea>
                      </div>
                    ) : (
                      <div className="p-8 text-center text-slate-500">
                        <Plane className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>No flight data available</p>
                        <Button 
                          onClick={fetchFlightData}
                          variant="outline"
                          className="mt-4"
                          size="sm"
                        >
                          Load Data
                        </Button>
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="summary" className="mt-4">
                    {flightData ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                            <div className="text-xs text-slate-400 mb-1">Latest Update</div>
                            <div className="text-lg font-semibold text-blue-400">
                              {formatTimestamp(flightData.latest_update)}
                            </div>
                          </div>
                         
                          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                            <div className="text-xs text-slate-400 mb-1">File Name</div>
                            <div className="text-sm font-mono text-slate-300">
                              {flightData.file_name}
                            </div>
                          </div>
                          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                            <div className="text-xs text-slate-400 mb-1">Status</div>
                            <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">
                              {flightData.status}
                            </Badge>
                          </div>
                        </div>

                        <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                          <h4 className="text-sm font-semibold text-slate-300 mb-2">Data Pipeline Flow:</h4>
                          <div className="text-xs text-slate-400 space-y-1">
                            <div>✓ OpenSky Network API → Real-time flight positions</div>
                            <div>✓ Azure Blob Storage → Raw JSON data stored</div>
                            <div>✓ Data Flow Transform → Geography enrichment</div>
                            <div>✓ Processed → Parquet format with Snappy compression</div>
                            <div>✓ Ready for analysis and visualization</div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="p-8 text-center text-slate-500">
                        <Plane className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>Waiting for pipeline data...</p>
                        <Button 
                          onClick={fetchFlightData}
                          variant="outline"
                          className="mt-4"
                          size="sm"
                        >
                          Load Data
                        </Button>
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="runs" className="mt-4">
                    <div className="rounded-lg border border-slate-700 overflow-hidden">
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead className="bg-slate-800">
                            <tr>
                              <th className="px-4 py-3 text-left text-slate-300 font-semibold">Run ID</th>
                              <th className="px-4 py-3 text-left text-slate-300 font-semibold">Status</th>
                              <th className="px-4 py-3 text-left text-slate-300 font-semibold">Start Time</th>
                              <th className="px-4 py-3 text-left text-slate-300 font-semibold">Duration</th>
                            </tr>
                          </thead>
                          <tbody className="bg-slate-900/50">
                            {pipelineRuns.map((run) => (
                              <tr key={run.runId} className="border-t border-slate-800">
                                <td className="px-4 py-3 text-slate-300 font-mono text-xs">
                                  {run.runId}
                                </td>
                                <td className="px-4 py-3">
                                  <div className="flex items-center gap-2">
                                    {getStatusIcon(run.status)}
                                    <span className="text-slate-300">{run.status}</span>
                                  </div>
                                </td>
                                <td className="px-4 py-3 text-slate-300">
                                  {formatTimestamp(run.startTime)}
                                </td>
                                <td className="px-4 py-3 text-slate-300">
                                  {run.duration ? `${run.duration}s` : "-"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </Card>
              
              <Card className="p-6 bg-slate-900 border-slate-800">
                <h2 className="text-xl font-semibold mb-4 text-white">Pipeline builder (does not affect live pipeline)</h2>
                <div className="space-y-3">
                  {activities.map((activity, index) => {
                    const activityType = activityTypes.find((t) => t.value === activity.type)
                    return (
                      <div key={activity.id}>
                        <div
                          draggable
                          onDragStart={() => handleDragStart(activity.id)}
                          onDragOver={(e) => handleDragOver(e, activity.id)}
                          onDragEnd={handleDragEnd}
                          className={`p-4 bg-slate-800 rounded-lg border-2 transition-all cursor-move ${
                            draggedActivity === activity.id 
                              ? "border-blue-500 opacity-50" 
                              : "border-transparent hover:border-slate-700"
                          }`}
                        >
                          <div className="flex items-start gap-3">
                            <GripVertical className="h-5 w-5 text-slate-400 mt-1 flex-shrink-0" />
                            <div className={`w-1 h-full ${activityType?.color} rounded-full flex-shrink-0`} />
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <Badge variant="secondary" className="text-xs">{index + 1}</Badge>
                                  <span className="text-lg">{activityType?.icon}</span>
                                  <span className="font-semibold text-white">{activity.name}</span>
                                </div>
                                <div className="flex gap-1">
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => duplicateActivity(activity)}
                                    className="h-8 w-8 p-0"
                                  >
                                    <Copy className="h-4 w-4 text-white" />
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => removeActivity(activity.id)}
                                    className="h-8 w-8 p-0 text-red-400"
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </div>
                              </div>
                              <p className="text-sm text-slate-400">{activity.description}</p>
                              <Badge variant="outline" className="border-slate-700 mt-2 text-white text-xs">
                                {activity.activityType}
                              </Badge>
                            </div>
                          </div>
                        </div>
                        {index < activities.length - 1 && (
                          <div className="flex justify-center py-2">
                            <div className="w-px h-6 bg-slate-700" />
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </Card>              
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}