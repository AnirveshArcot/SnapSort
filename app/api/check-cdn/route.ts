import { NextResponse } from "next/server";

export async function GET() {
  try {
    const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/status/cdn-mounted`, {
      cache: "no-store",
    });

    if (!res.ok) {
      throw new Error("Failed to fetch from backend");
    }

    const data = await res.json();
    return NextResponse.json({ mounted: data.mounted });
  } catch (error) {
    console.error("Failed to fetch CDN status from FastAPI:", error);
    return NextResponse.json({ mounted: false }, { status: 500 });
  }
}