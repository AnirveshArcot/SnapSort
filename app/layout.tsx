import type React from "react";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";
import { redirect } from "next/navigation";

export const metadata = {
  title: "SnapSort - Share Event Photos",
};

async function checkCDNMounted(): Promise<boolean> {
  try {
    const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}/api/check-cdn`, {
      cache: "no-store",
    });
    const data = await res.json();
    return data.mounted;
  } catch (err) {
    console.error("CDN check failed:", err);
    return false;
  }
}

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const cdnMounted = await checkCDNMounted();

  if (!cdnMounted) {
    redirect("/cdn-unavailable");
  }

  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background font-sans antialiased">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="relative flex min-h-screen flex-col">
            <main className="flex-1">{children}</main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
