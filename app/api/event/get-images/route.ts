import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const cookie = req.headers.get("cookie") ?? "";

  const { searchParams } = new URL(req.url);
  const skip = searchParams.get("skip") ?? "0";
  const limit = searchParams.get("limit") ?? "20";

  try {
    const backendRes = await fetch(
      `${process.env.BACKEND_URL}/event/get-images?skip=${skip}&limit=${limit}`,
      {
        headers: { cookie },
      }
    );

    if (!backendRes.ok) {
      return NextResponse.json({ images: [], total_count: 0 }, { status: backendRes.status });
    }

    const data = await backendRes.json();
    return NextResponse.json({
      images: data.images || [],
      total_count: data.total_count ?? 0,
    });
  } catch (err) {
    console.error("Failed to fetch images from backend:", err);
    return NextResponse.json({ images: [], total_count: 0 }, { status: 500 });
  }
}
