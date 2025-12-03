import Link from "next/link"
import Image from "next/image"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Navigation } from "@/components/navigation"
import { Info } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden">
      <Navigation />
      {/* Subtle texture overlay */}
      <div className="absolute inset-0 opacity-[0.025] pointer-events-none bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48ZmlsdGVyIGlkPSJhIiB4PSIwIiB5PSIwIj48ZmVUdXJidWxlbmNlIGJhc2VGcmVxdWVuY3k9Ii43NSIgc3RpdGNoVGlsZXM9InN0aXRjaCIgdHlwZT0iZnJhY3RhbE5vaXNlIi8+PGZlQ29sb3JNYXRyaXggdHlwZT0ic2F0dXJhdGUiIHZhbHVlcz0iMCIvPjwvZmlsdGVyPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbHRlcj0idXJsKCNhKSIvPjwvc3ZnPg==')]" />

      {/* Azure background image - original dimensions */}
      <div className="fixed bottom-0 left-0 w-[1100px] h-[1100px] -translate-x-1/3 translate-y-1/3 opacity-[0.20] pointer-events-none z-0">
        <Image
          src="/Microsoft-Azure.png"
          alt="Machine Learning Illustration"
          width={1100}
          height={1100}
          className="w-full h-full object-contain"
        />
      </div>

      {/* Gradient overlay */}

      <div className="relative pt-32 pb-10 md:pb-1 px-4">
        <div className="container mx-auto max-w-6xl">
          {/* Hero Section - minimalist style */}
          <div className="text-center mb-10">
            <h1 className="text-5xl md:text-7xl font-light mb-6 text-white tracking-tight leading-tight text-balance">
              <span className="text-blue-500 font-bold">Azure </span>AI
              <span className="block mt-2 font-light">Learning Playground</span>
            </h1>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-8 leading-relaxed text-pretty font-light">
              Learn the fundamentals of <span className="font-semibold text-blue-500">Natural Language Processing</span>, 
              <span className="font-semibold text-blue-500"> Computer Vision</span>, and <span className="font-semibold text-blue-500">Pipeline Orchestration </span> 
              using Azure's infrastructure.
              <span className="font-normal text-blue-500"> Note:</span> during training, avoid changing tabs or refreshing the page. Training typically takes ~8 minutes on an f4s_v2 VM.
            </p>
          </div>

          <div className="max-w-3xl mx-auto mb-12rounded-lg">
            <Card className="p-6 bg-transparent border-gray-800 backdrop-blur-sm">
              <div className="flex items-center justify-center gap-4">
                <div className="text-center">
                  <p className="text-lg md:text-xl text-white mb-10 font-medium">
                    Look for these info icons throughout the pages
                  </p>
                  <div className="flex items-center justify-center gap-3">
                    <TooltipProvider delayDuration={300}>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="p-3 bg-slate-900 rounded-full border-2 border-gray-500 cursor-help hover:bg-slate-800 transition-colors">
                            <Info className="h-6 w-6 text-gray-400 animate-pulse" />
                          </div>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          <p className="text-sm font-semibold">I am a tooltip! 🎉</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <span className="text-slate-300 text-base">← Hover over me!</span>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Training Options Grid - original grid structure */}
          <div className="grid p-10 md:p-0 md:grid-cols-3 gap-6 mb-16 mt-10">
            {/* Image Classifier Card */}
            <Link href="/cnn" className="group">
              <Card className="p-6 bg-slate-900/50 border-white/5 hover:border-blue-500 transition-all cursor-pointer backdrop-blur-sm relative overflow-hidden h-full">
                <div className="absolute inset-0 bg-gradient-to-br from-[#c9b58c]/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="relative pb-4">
                  <h3 className="text-xl font-light mb-3 text-white group-hover:text-blue-500 transition-colors">
                    Computer Vision
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      ShuffleNet
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      MobileNet
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      Classification
                    </Badge>
                  </div>
                </div>
              </Card>
            </Link>

            {/* Spam Detection Card */}
            <Link href="/train-spam" className="group">
              <Card className="p-6 bg-slate-900/50 border-white/5 hover:border-blue-500 transition-all cursor-pointer backdrop-blur-sm relative overflow-hidden h-full">
                <div className="absolute inset-0 bg-gradient-to-br from-[#c9b58c]/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="relative pb-4">
                  <h3 className="text-xl font-light mb-3 text-white group-hover:text-blue-500 transition-colors">
                    Natural Language Processing
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      DistilBert
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      Tokenizer
                    </Badge>
                  </div>
                </div>
              </Card>
            </Link>

            {/* Pipeline Builder Card */}
            <Link href="/pipeline" className="group">
              <Card className="p-6 bg-slate-900/50 border-white/5 hover:border-blue-500 transition-all cursor-pointer backdrop-blur-sm relative overflow-hidden h-full">
                <div className="absolute inset-0 bg-gradient-to-br from-[#c9b58c]/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="relative pb-4">
                  <h3 className="text-xl font-light mb-3 text-white group-hover:text-blue-500 transition-colors">
                    Pipeline Basics
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      Orchestration
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      MLOps
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="text-xs bg-white/5 border border-white/10 text-slate-300 font-light"
                    >
                      Automation
                    </Badge>
                  </div>
                </div>
              </Card>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
