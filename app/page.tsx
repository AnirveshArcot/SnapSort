"use client";

import { useRouter } from "next/navigation";
import { getSession, getImages, downloadImageBlob } from "@/lib/api";
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
  name: string;
  base64: string;
}

export default function EventImagesPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [user, setUser] = useState<User | null>(null);
  const [images, setImages] = useState<EventImage[]>([]);
  const [loadingImages, setLoadingImages] = useState(true);

  const handleDownload = async (filename: string) => {
    try {
      const blob = await downloadImageBlob(filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const modifiedName = filename.replace("_preview", "");
      a.download = modifiedName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download failed:", err);
      alert("Failed to download image.");
    }
  };

  useEffect(() => {
    (async () => {
      const sessionUser = await getSession();
      if (sessionUser) {
        // Redirect admins/editors/photographers to the admin page
        if (
          sessionUser.role === "admin" ||
          sessionUser.role === "photographer" ||
          sessionUser.role === "editor"
        ) {
          router.push("/admin");
          return;
        }
        setUser(sessionUser);
        try {
          const res = await getImages();
          setImages(Array.isArray(res) ? res : res.images || []);
        } catch (err) {
          setImages([]);
        }
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
            <span className="text-xl font-bold">SnapSort</span>
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
      {loadingImages ? (
        <div className="flex items-center justify-center h-screen">
          <p>Loading Images....</p>
        </div>
      ) : images.length === 0 ? (
        <div className="text-center py-12 border rounded-lg mx-4 sm:mx-10">
          <p className="text-muted-foreground">
            No images have been uploaded to this event yet.
          </p>
        </div>
      ) : (
        <div className="space-y-2 px-3 sm:px-10">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {images.map((img) => (
              <div
                key={img.name}
                className="border rounded-lg overflow-hidden shadow hover:shadow-md transition"
              >
                <img
                  src={img.base64}
                  alt={img.name}
                  className="w-full h-48 object-cover"
                />
                <div className="p-2 flex justify-between items-center text-sm">
                  <span className="truncate" title={img.name}>{img.name}</span>
                  <Button onClick={() => handleDownload(img.name)}>
                    Download
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
