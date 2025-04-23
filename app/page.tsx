"use client";

import { useRouter } from "next/navigation";
import Image from "next/image";
import { getEventImages, getSession } from "@/lib/api";
import { useEffect, useState } from "react";
import Link from "next/link";
import { UserNav } from "@/components/user-nav";
import { Button } from "@/components/ui/button";

interface User {
  id: string;
  name: string;
  email: string;
  password: string | null;
  image: string;
  joined_event: string;
  role: string;
}

interface EventImage {
  id: number;
  image_base64: string;
}

export default function EventImagesPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [user, setUser] = useState<User | null>(null);
  const [images, setImages] = useState<EventImage[]>([]);
  const [loadingImages, setLoadingImages] = useState(true);

  useEffect(() => {
    (async () => {
      const sessionUser = await getSession();

      if (sessionUser) {
        if(sessionUser.role=='admin'){
          router.push("/admin");
          return;
        }
        setUser(sessionUser);
        const imgs = await getEventImages();
        setImages(imgs);
        setLoadingImages(false);
        setChecking(false);
      } else {
        router.push("/login");
      }
    })();
  }, [router]);

  if (checking) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p>Checking authentication…</p>
      </div>
    );
  }

  return (
    <div className="">
      <header className="border-b">
        <div className="flex h-16 items-center justify-between px-3 sm:px-10">
          <Link href="/" className="flex items-center">
            <span className="text-xl font-bold">EventShare</span>
          </Link>

          <nav className="flex items-center">
            {user ? (
              <UserNav user={user} />
            ) : (
              <div className="flex items-center gap-2">
                <Link href="/login">
                  <Button variant="ghost">Login</Button>
                </Link>
                <Link href="/register">
                  <Button>Register</Button>
                </Link>
              </div>
            )}
          </nav>
        </div>
      </header>

      <h2 className="text-2xl font-semibold my-4 px-3 sm:px-10">Event Images</h2>
      {loadingImages === true ? (<div className="flex items-center justify-center h-screen">
        <p>Loading Images....</p>
      </div>): (images.length === 0 ? (
        <div className="text-center py-12 border rounded-lg">
          <p className="text-muted-foreground">
            No images have been uploaded to this event yet.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 px-3 sm:px-10">
          {images.map((image) => (
            <div key={image.id} className="border rounded-lg overflow-hidden">
              <div className="relative aspect-square">
                <Image
                  src={image.image_base64}
                  alt={`Image`}
                  fill
                  className="object-cover"
                />
              </div>
            </div>
          ))}
        </div>
      ))}
      
    </div>
  );
}
