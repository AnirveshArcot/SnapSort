"use client";

import type React from "react";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { registerUser } from "@/lib/api";

export function RegisterForm() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const router = useRouter();

  function handleImageChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] || null;
    setImageFile(file);

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setIsLoading(true);
    setError("");
  
    const formData = new FormData(e.currentTarget);
    const userData: Record<string, any> = {
      name: formData.get("name"),
      email: formData.get("email"),
      password: formData.get("password"),
      image: null, // Default image to null
    };
  
    if (imageFile) {
      const reader = new FileReader();
      reader.readAsDataURL(imageFile);
      reader.onloadend = async () => {
        userData.image = reader.result as string;
  
        try {
          const result = await registerUser(userData); // registerUser should send JSON
          if (result.error) {
            setError(result.error);
          } else {
            router.push("/login");
          }
        } catch (err) {
          setError("Registration failed. Please try again.");
        } finally {
          setIsLoading(false);
        }
      };
    } else {
      try {
        const result = await registerUser(userData); // registerUser should send JSON
        if (result.error) {
          setError(result.error);
        } else {
          router.push("/login");
        }
      } catch (err) {
        setError("Registration failed. Please try again.");
      } finally {
        setIsLoading(false);
      }
    }
  }
  

  return (
    <div className="grid gap-6">
      <form onSubmit={handleSubmit}>
        <div className="grid gap-4">
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
            <Label htmlFor="profile-image">Profile Image</Label>
            <div className="flex items-center gap-4">
              <Input
                id="profile-image"
                name="profile-image"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                disabled={isLoading}
              />
              {imagePreview && (
                <div className="relative h-12 w-12 rounded-full overflow-hidden">
                  <img
                    src={imagePreview || "/placeholder.svg"}
                    alt="Profile preview"
                    className="h-full w-full object-cover"
                  />
                </div>
              )}
            </div>
          </div>

          {error && <p className="text-sm text-destructive">{error}</p>}

          <Button type="submit" disabled={isLoading} className="w-full">
            {isLoading ? "Creating account..." : "Create account"}
          </Button>
        </div>
      </form>
    </div>
  );
}
