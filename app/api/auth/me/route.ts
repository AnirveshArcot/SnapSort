
import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const cookie = req.headers.get("cookie") ?? "";

  let backendRes: Response;
  try {
    backendRes = await fetch(`${process.env.BACKEND_URL}/auth/me`, {
      headers: { cookie },
    });
  } catch (err) {
    return NextResponse.json(null);
  }

  if (!backendRes.ok) {
    return NextResponse.json(null);
  }

  const data = await backendRes.json().catch(() => ({}));
  return NextResponse.json(data);
}
