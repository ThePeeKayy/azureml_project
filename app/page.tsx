import Link from "next/link"
import Image from 'next/image'
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Navigation } from "@/components/navigation"
import { ArrowRight, ImageIcon, Mail, Workflow, Sparkles, Database, Zap } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden">
      <div className="absolute inset-0 opacity-[0.025] pointer-events-none bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48ZmlsdGVyIGlkPSJhIiB4PSIwIiB5PSIwIj48ZmVUdXJidWxlbmNlIGJhc2VGcmVxdWVuY3k9Ii43NSIgc3RpdGNoVGlsZXM9InN0aXRjaCIgdHlwZT0iZnJhY3RhbE5vaXNlIi8+PGZlQ29sb3JNYXRyaXggdHlwZT0ic2F0dXJhdGUiIHZhbHVlcz0iMCIvPjwvZmlsdGVyPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbHRlcj0idXJsKCNhKSIvPjwvc3ZnPg==')]" />

      <div className="fixed bottom-0 left-0 w-[1100px] h-[1100px] -translate-x-1/3 translate-y-1/3 opacity-[0.20] pointer-events-none z-0">
        <Image
          src="/Microsoft-Azure.png"
          alt="Machine Learning Illustration"
          width={1100}
          height={1100}
          className="w-full h-full object-contain"
        />
      </div>

      <div className="absolute inset-0 bg-gradient-to-br from-blue-950/10 via-transparent to-slate-950/50 pointer-events-none" />

      <Navigation />

      <div className="relative pt-32 pb-10 md:pb-1 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center mb-10">
            <h1 className="text-5xl md:text-7xl font-bold mb-6 text-white tracking-tight leading-tight text-balance">
              Machine Learning
              <span className="block bg-gradient-to-r from-blue-400 via-blue-500 to-cyan-400 bg-clip-text text-transparent mt-2">
                Training Platform
              </span>
            </h1>

            <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-8 leading-relaxed text-pretty">
              Train real models and observe real pipelines powered by Azure infrastructure.

              <span className="font-bold"> Disclaimer:</span> you cannot change tabs or refresh the page during training. Training takes ~8mins on an f4s_v2 VM.

            </p>

          </div>

          <div className="grid p-10 md:p-0 md:grid-cols-3 gap-6 mb-16">
            {/* Image Classifier Card */}
            <Link href="/cnn" className="group">
              <Card className="p-6 bg-slate-900/50 border-white/5 hover:border-blue-500/30 transition-all cursor-pointer backdrop-blur-sm relative overflow-hidden h-full">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                <div className="relative pb-4">
                  <ImageIcon className="h-[90%] w-[90%] text-blue-400" />
                 

                  <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-blue-400 transition-colors">
                    Image Classifier
                  </h3>

                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      CNN
                    </Badge>
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      Transfer Learning
                    </Badge>
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      Real-time
                    </Badge>
                  </div>
                </div>
              </Card>
            </Link>

            {/* Spam Detection Card */}
            <Link href="/train-spam" className="group">
              <Card className="p-6 bg-slate-900/50 border-white/5 hover:border-emerald-500/30 transition-all cursor-pointer backdrop-blur-sm relative overflow-hidden h-full">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                <div className="relative pb-4">
                  <Mail className="h-[90%] w-[90%] text-emerald-400" />

                  <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-emerald-400 transition-colors">
                    Spam Detector
                  </h3>

                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      NLP
                    </Badge>
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      Classification
                    </Badge>
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      TF-IDF
                    </Badge>
                  </div>
                </div>
              </Card>
            </Link>

            {/* Pipeline Builder Card */}
            <Link href="/pipeline" className="group">
              <Card className="p-6 bg-slate-900/50 border-white/5 hover:border-purple-500/30 transition-all cursor-pointer backdrop-blur-sm relative overflow-hidden h-full">
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                <div className="relative pb-4">
                  <Workflow className="h-[90%] w-[90%] text-purple-400" />

                  <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-purple-400 transition-colors">
                    Pipeline Builder
                  </h3>


                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      Orchestration
                    </Badge>
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
                      MLOps
                    </Badge>
                    <Badge variant="secondary" className="text-xs bg-white/5 border border-white/10 text-slate-300">
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
