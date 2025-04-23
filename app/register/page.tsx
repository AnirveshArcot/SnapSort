"use client";
import Link from "next/link"
import { useRouter } from "next/navigation"
import { RegisterForm } from "@/components/register-form"
import { getSession } from "@/lib/api"
import { useEffect, useState } from "react";

export default function RegisterPage() {
  const [checking, setChecking] = useState(true);
  const router = useRouter();
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
          <h1 className="text-2xl font-semibold tracking-tight">Create an account</h1>
          <p className="text-sm text-muted-foreground">Enter your details below to create your account</p>
        </div>
        <RegisterForm />
        <p className="px-8 text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <Link href="/login" className="underline underline-offset-4 hover:text-primary">
            Login
          </Link>
        </p>
      </div>
    </div>
  )
}

