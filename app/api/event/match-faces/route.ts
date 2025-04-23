import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL;

export async function POST(request: Request) {
  const cookie = request.headers.get("cookie") ?? "";
  const bodyText = await request.text(); 

  const res = await fetch(`${BACKEND_URL}/admin/match_faces`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      cookie,
    },
    body: bodyText,
  });

  const payload = await res.json();
  if (!res.ok) {
    return NextResponse.json(
      { error: payload.detail || payload.error || "Unknown error" },
      { status: res.status }
    );
  }
  return NextResponse.json(payload);
}
