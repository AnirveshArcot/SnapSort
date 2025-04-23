import { NextResponse } from "next/server";



export async function POST(request: Request) {
  const cookie = request.headers.get("cookie") ?? "";
  const res = await fetch(`${process.env.BACKEND_URL}/admin/process_user_images`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      cookie,
    },
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
