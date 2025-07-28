import type React from "react";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";
import { redirect } from "next/navigation";
import { checkCDNMounted } from "@/lib/api";

export const metadata = {
  title: "SnapSort - Share Event Photos",
};

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
