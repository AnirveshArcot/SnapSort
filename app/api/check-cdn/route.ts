import { NextResponse } from "next/server";

export async function GET() {
  try {
    const res = await fetch(`${process.env.BACKEND_URL}/status/cdn-mounted`, {
      cache: "no-store",
    });
    const data = await res.json();

    return NextResponse.json({ mounted: data.mounted });
  } catch (error) {
    console.error("Failed to fetch CDN status from FastAPI:", error);
    return NextResponse.json({ mounted: false }, { status: 500 });
  }
}
