import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const filename = req.nextUrl.searchParams.get("filename");
  if (!filename) return NextResponse.json({ error: "Missing filename" }, { status: 400 });

  const cookie = req.headers.get("cookie") ?? "";

  const backendRes = await fetch(`${process.env.BACKEND_URL}/event/download?filename=${encodeURIComponent(filename)}`, {
    headers: { cookie },
  });

  if (!backendRes.ok) {
    return NextResponse.json({ error: "Failed to fetch file" }, { status: backendRes.status });
  }

  const blob = await backendRes.blob();

  return new Response(blob, {
    headers: {
      "Content-Type": backendRes.headers.get("Content-Type") || "application/octet-stream",
      "Content-Disposition": `attachment; filename="${filename.replace('_preview', '_original')}"`,
    },
  });
}