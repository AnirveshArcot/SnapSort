import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL;  // e.g. "http://localhost:8000"

export async function POST(request: Request) {
  const cookie = request.headers.get("cookie") ?? "";
  const body = await request.text();  // the JSON payload from the client

  const res = await fetch(`${BACKEND_URL}/event/upload-images`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      cookie,
    },
    body,
  });

  const payload = await res.json();
  if (!res.ok) {
    return NextResponse.json(
      { error: payload.detail || payload.error || "Upload failed" },
      { status: res.status }
    );
  }
  return NextResponse.json(payload, { status: 201 });
}
