"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { registerUser } from "@/lib/api"; // Renamed import

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  onClose: () => void;
}

function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!window.isSecureContext) {
      setError("Camera works only on HTTPS or localhost.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera API not supported in this browser.");
      return;
    }

    let active = true;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        if (!active) return;
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err: any) {
        setError("Unable to access camera: " + (err?.message || "Permission denied."));
      }
    })();

    return () => {
      active = false;
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  const takePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], "camera.jpg", { type: "image/jpeg" });
      onCapture(file);
      onClose();
    }, "image/jpeg");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="flex flex-col items-center gap-4 rounded-xl bg-background p-4 shadow-xl">
        {error ? (
          <p className="max-w-xs text-center text-sm text-destructive">{error}</p>
        ) : (
          <video
            ref={videoRef}
            playsInline
            autoPlay
            muted
            className="h-auto max-w-full rounded-lg transform -scale-x-100"
          />
        )}
        <div className="mt-2 flex gap-4">
          <Button onClick={takePhoto} disabled={!!error}>Take photo</Button>
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
        </div>
      </div>
    </div>
  );
}

export function RegisterForm() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(false);
  const router = useRouter();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setImageFile(file);
    setImagePreview(file ? URL.createObjectURL(file) : null);
  };

  const handleCapture = (file: File) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    const formData = new FormData(e.currentTarget);
    if (imageFile) {
      formData.set("image", imageFile); // Important: must match FastAPI param
    }

    try {
      const response = await registerUser(formData);
      router.push("/login");
    } catch (err: any) {
      setError(err.message || "Registration failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid gap-6">
      {showCamera && <CameraCapture onCapture={handleCapture} onClose={() => setShowCamera(false)} />}
      <form onSubmit={handleSubmit} className="grid gap-4" encType="multipart/form-data">
        <div className="grid gap-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" name="name" placeholder="John Doe" required disabled={isLoading} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="email">Email</Label>
          <Input id="email" name="email" type="email" placeholder="name@example.com" required disabled={isLoading} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="password">Password</Label>
          <Input id="password" name="password" type="password" required disabled={isLoading} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="image">Profile Image</Label>
          <div className="flex items-center gap-4">
            <Input
              id="image"
              name="image"
              type="file"
              accept="image/*"
              capture="user"
              onChange={handleFileChange}
              disabled={isLoading}
            />
            <Button type="button" onClick={() => setShowCamera(true)} disabled={isLoading}>
              Use camera
            </Button>
            {imagePreview && (
              <div className="relative h-12 w-12 overflow-hidden rounded-full">
                <img src={imagePreview} alt="Preview" className="h-full w-full object-cover" />
              </div>
            )}
          </div>
          <p className="text-xs text-muted-foreground">For best results, take your photo under proper lighting.</p>
        </div>
        <div className="flex items-start gap-2">
          <input id="terms" name="terms" type="checkbox" required className="mt-1 h-4 w-4" />
          <Label htmlFor="terms" className="text-sm">
            I agree to the{" "}
            <a href="/terms-indian-penal-code" className="underline" target="_blank" rel="noopener noreferrer">
              Terms & Conditions
            </a>
          </Label>
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Button type="submit" disabled={isLoading} className="w-full">
          {isLoading ? "Creating account…" : "Create account"}
        </Button>
      </form>
    </div>
  );
}
