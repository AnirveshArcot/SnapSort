"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { uploadEventImage } from "@/lib/actions"

interface ImageUploadFormProps {
  eventId: string
  userId: string
}

export function ImageUploadForm({ eventId, userId }: ImageUploadFormProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const router = useRouter()

  function handleImageChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0] || null
    setImageFile(file)

    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    } else {
      setImagePreview(null)
    }
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()

    if (!imageFile) {
      setError("Please select an image to upload")
      return
    }

    setIsLoading(true)
    setError("")

    const formData = new FormData()
    formData.append("image", imageFile)
    formData.append("eventId", eventId)
    formData.append("userId", userId)

    try {
      const result = await uploadEventImage(formData)

      if (result.error) {
        setError(result.error)
      } else {
        setImageFile(null)
        setImagePreview(null)
        router.refresh()
      }
    } catch (err) {
      setError("Failed to upload image. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="border rounded-lg p-4">
      <h3 className="text-lg font-medium mb-4">Upload a new image</h3>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
          <div className="flex-1">
            <input
              type="file"
              accept="image/*"
              id="event-image"
              className="hidden"
              onChange={handleImageChange}
              disabled={isLoading}
            />
            <label
              htmlFor="event-image"
              className="flex h-32 w-full cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-gray-300 hover:bg-gray-50"
            >
              {imagePreview ? (
                <div className="relative h-full w-full">
                  <img
                    src={imagePreview || "/placeholder.svg"}
                    alt="Preview"
                    className="h-full w-full object-contain"
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="h-6 w-6 text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground">Click to upload an image</p>
                </div>
              )}
            </label>
          </div>
          <Button type="submit" disabled={isLoading || !imageFile} className="w-full sm:w-auto">
            {isLoading ? "Uploading..." : "Upload Image"}
          </Button>
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
      </form>
    </div>
  )
}

