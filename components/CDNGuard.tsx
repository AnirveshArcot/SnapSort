"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { checkCDNMounted } from "@/lib/api";

export default function CDNGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  useEffect(() => {
    const verifyCDN = async () => {
      try {
        const data = await checkCDNMounted();
        if (!data.mounted) {
          router.replace("/cdn-unavailable");
        }
      } catch (err) {
        console.error("CDN check failed:", err);
        router.replace("/cdn-unavailable");
      }
    };

    verifyCDN();
  }, []);

  return <>{children}</>;
}
