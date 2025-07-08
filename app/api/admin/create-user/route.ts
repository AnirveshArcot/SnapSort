import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  const res = await fetch(`${backendUrl}/admin/create-user`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(req.headers.get('cookie') ? { 'cookie': req.headers.get('cookie')! } : {}),
    },
    body: JSON.stringify(body),
    credentials: 'include',
  });
  const data = await res.json();
  return new NextResponse(JSON.stringify(data), {
    status: res.status,
    headers: { 'Content-Type': 'application/json' },
  });
} 