import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL;

export async function POST(request: Request) {
  const cookie = request.headers.get("cookie") ?? "";
  const formData = await request.formData();

  const res = await fetch(`${BACKEND_URL}/auth/register`, {
    method: "POST",
    headers: {
      cookie,
    },
    body: formData,
  });

  const payload = await res.json();

  if (!res.ok) {
    return NextResponse.json(
      { error: payload.detail || payload.error || "Registration failed" },
      { status: res.status }
    );
  }

  const response = NextResponse.json(payload, { status: 201 });

  const setCookie = res.headers.get("set-cookie");
  if (setCookie) {
    response.headers.set("set-cookie", setCookie);
  }

  return response;
}
