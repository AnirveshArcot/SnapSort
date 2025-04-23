import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  const body = await req.json()
  const cookie = req.headers.get('cookie') ?? ''


  const backendRes = await fetch(
    `${process.env.BACKEND_URL}/auth/register`,
    {
      method: 'POST',
      headers: {
        cookie,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }
  )

  const data = await backendRes.json()
  const res = NextResponse.json(data, { status: backendRes.status })

  const setCookie = backendRes.headers.get('set-cookie')
  if (setCookie) {
    res.headers.set('set-cookie', setCookie)
  }

  return res
}
