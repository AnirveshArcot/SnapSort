import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const cookie = req.headers.get("cookie") ?? "";

  let backendRes: Response;
  try {
    backendRes = await fetch(`${process.env.BACKEND_URL}/event/images`, {
      method: "GET",
      headers: {
        cookie,
      },
    });
  } catch (err) {
    console.error("Failed to fetch from backend:", err);
    return NextResponse.json({ images: [] }, { status: 502 });
  }

  if (!backendRes.ok) {
    return NextResponse.json({ images: [] }, { status: backendRes.status });
  }

  const data = await backendRes.json().catch(() => ({ images: [] }));
  return NextResponse.json(data);
}
