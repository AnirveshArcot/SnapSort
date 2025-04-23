
import { NextResponse } from 'next/server'

export async function POST(req: Request) {

  const cookie = req.headers.get('cookie') ?? ''
  const backendRes = await fetch(`http://localhost:8000/auth/logout`, {
    method: 'POST',
    headers: { cookie },
  })

  if (!backendRes.ok) {
    return NextResponse.json(
      { error: 'Logout failed' },
      { status: backendRes.status }
    )
  }


  const res = NextResponse.json({ message: 'Logged out' })
  res.cookies.set('auth_token', '', {
    httpOnly: true,
    secure: false,
    sameSite: 'lax',
    expires: new Date(0),
  })
  return res
}
