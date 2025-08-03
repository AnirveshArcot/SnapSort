import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const cookie = req.headers.get("cookie") ?? "";

  const backendRes = await fetch(`${process.env.BACKEND_URL}/register`, {
    method: "POST",
    headers: {
      cookie,
    },
    body: formData,
  });

  const data = await backendRes.json();
  const res = NextResponse.json(data, { status: backendRes.status });

  const setCookie = backendRes.headers.get("set-cookie");
  if (setCookie) {
    res.headers.set("set-cookie", setCookie);
  }

  return res;
}
