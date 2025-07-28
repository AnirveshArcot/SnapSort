"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";

export default function CDNGuard({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    const checkCDNMounted = async () => {
      try {
        const res = await fetch("/api/cdn-status", { cache: "no-store" });
        const data = await res.json();
        if (!data.ok) {
          router.push("/login");
        }
      } catch (err) {
        console.error("CDN check failed:", err);
        router.push("/login");
      }
    };

    checkCDNMounted();
  }, [pathname]);

  return <>{children}</>;
}
