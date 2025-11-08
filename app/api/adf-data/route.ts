import { NextResponse } from "next/server"

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const endpoint = url.searchParams.get("endpoint") || ""

    const base = process.env.NEXT_PUBLIC_ADF_API_URL
    if (!base) throw new Error("NEXT_PUBLIC_ADF_API_URL not configured")

    const res = await fetch(`${base}/${endpoint}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })

    if (!res.ok) throw new Error(`Azure Function failed: ${res.status}`)

    const data = await res.json()

    return NextResponse.json(data)

  } catch (err: any) {
    return NextResponse.json(
      { error: err.message || "Internal server error" },
      { status: 500 }
    )
  }
}
