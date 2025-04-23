"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import { UserNav } from "@/components/user-nav";
import { Button } from "@/components/ui/button";
import { getSession, uploadEventImages } from "@/lib/api";
import { createNewEvent, matchFaces, generateFeatures } from "@/lib/api";


interface User {
  id: string;
  name: string;
  email: string;
  password: string | null;
  image: string;
  joined_event: string;
  role: string;
}

export default function AdminPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [user, setUser] = useState<User | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  useEffect(() => {
    (async () => {
      const sessionUser = await getSession();

      if (sessionUser) {
        if (sessionUser.role !== "admin") {
          router.push("/");
          return;
        }
        setUser(sessionUser);
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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

  const convertToBase64 = (file: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
    });

    const handleUpload = async () => {
      try {
        const base64Images = await Promise.all(
          selectedFiles.map((file) => convertToBase64(file))
        );
  
        const { uploaded } = await uploadEventImages(base64Images);
        console.log(`Images uploaded successfully! (${uploaded.length} files)`);
        setSelectedFiles([]);
      } catch (err: any) {
        console.error(err);
        alert(`Upload failed: ${err.message || err}`);
      }
    };

  const handleCreateEvent = async () => {
    if (!confirm("Are you sure you want to create a new event?")) return;

    try {
      const data = await createNewEvent();
      console.log(`Event created with ID: ${data.event_id}`);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleMatchFaces = async () => {
    try {
      const data = await matchFaces(/* any request body here, if needed */);
      console.log(`Matching complet`);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleGenerateFeatures = async () => {
    try {
      const data = await generateFeatures();
      console.log(`Feature vectors generated for images.`);
    } catch (err: any) {
      console.error(err);
    }
  };

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

      <h2 className="text-2xl font-semibold my-6 px-3 sm:px-10">Admin Panel</h2>

      <div className="px-3 sm:px-10 space-y-6">
        <div className="space-y-2">
          <label className="font-medium">Upload Images</label>
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleFileChange}
            className="block"
          />
          <Button onClick={handleUpload} disabled={!selectedFiles.length}>
            Upload Selected Images
          </Button>
        </div>

        {/* Action Buttons */}
        <div className="space-y-2">
          <Button onClick={handleCreateEvent}>Create New Event</Button>
          <Button onClick={handleMatchFaces} className="ml-4">
            Match Faces
          </Button>
          <Button onClick={handleGenerateFeatures} className="ml-4">
            Generate Features
          </Button>
        </div>
      </div>
    </div>
  );
}
