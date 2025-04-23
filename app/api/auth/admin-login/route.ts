import { NextResponse } from "next/server";

export async function POST(req: Request) {
    const rawBody = await req.text();                                       // read the URL‑encoded string
    const cookie  = req.headers.get('cookie') ?? ''
  
    const backendRes = await fetch(`${process.env.BACKEND_URL}/auth/admin/login`, {
      method: 'POST',
      headers: {
        cookie,
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
      },
      body: rawBody
    });
  
    const data = await backendRes.json().catch(() => ({}));
    const res  = NextResponse.json(data, { status: backendRes.status });
  
    const setCookie = backendRes.headers.get('set-cookie');
    if (setCookie) res.headers.set('set-cookie', setCookie);
  
    return res;
  }