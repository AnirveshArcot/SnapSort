// app/layout.tsx
import type React from "react";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";
import CDNGuard from "@/components/CDNGuard";

export const metadata = {
  title: "SnapSort - Share Event Photos",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background font-sans antialiased">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <CDNGuard>
            <div className="relative flex min-h-screen flex-col">
              <main className="flex-1">{children}</main>
            </div>
          </CDNGuard>
        </ThemeProvider>
      </body>
    </html>
  );
}
