"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { checkCDNMounted } from "@/lib/api";

export default function CDNGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const verifyCDN = async () => {
      try {
        const data = await checkCDNMounted();

        // Only redirect if the CDN is not mounted
        if (!data.mounted && pathname !== "/cdn-unavailable") {
          router.replace("/cdn-unavailable");
        }

        // Optional: if CDN is mounted and user is on /cdn-unavailable, send back to home
        if (data.mounted && pathname === "/cdn-unavailable") {
          router.replace("/");
        }
      } catch (err) {
        console.error("CDN check failed:", err);
        if (pathname !== "/cdn-unavailable") {
          router.replace("/cdn-unavailable");
        }
      }
    };

    verifyCDN();
  }, [pathname]);

  return <>{children}</>;
}
