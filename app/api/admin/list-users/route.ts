import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  const res = await fetch(`${backendUrl}/admin/list-users`, {
    method: 'GET',
    headers: {
      ...(req.headers.get('cookie') ? { 'cookie': req.headers.get('cookie')! } : {}),
    },
    credentials: 'include',
  });
  const data = await res.json();
  return new NextResponse(JSON.stringify(data), {
    status: res.status,
    headers: { 'Content-Type': 'application/json' },
  });
} 