'use client';
import Link from "next/link";
import { useEffect, useState } from "react";
import { LoginForm } from "@/components/login-form";
import { getSession } from "@/lib/api";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    ;(async () => {
      const user = await getSession()
      if (user) {
        router.replace('/')
      } else {
        setChecking(false)
      }
    })()
  }, [router])


  if (checking) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p>Checking authentication…</p>
      </div>
    )
  }

  return (
    <div className="flex h-screen w-screen flex-col items-center justify-center">
      <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px] px-10 sm:px-0">
        <div className="flex flex-col space-y-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">Login to your account</h1>
          <p className="text-sm text-muted-foreground">
            Enter your email and password below to login
          </p>
        </div>
        <LoginForm />
        <p className="px-8 text-center text-sm text-muted-foreground">
          Don't have an account?{" "}
          <Link href="/register" className="underline underline-offset-4 hover:text-primary">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
