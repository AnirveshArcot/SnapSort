"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import { UserNav } from "@/components/user-nav";
import { Button } from "@/components/ui/button";
import { uploadImages, getImages, downloadImageBlob, createUserAsAdmin, getAllUsersForAdmin, deleteUserAsAdmin, getSession, matchFaces, createNewEvent } from "@/lib/api";
import { toast } from "sonner";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import pLimit from "p-limit";

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
  const [uploading, setUploading] = useState(false);
  const [imageList, setImageList] = useState<{ name: string; base64: string }[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  // --- Manage Users State ---
  const [newUserName, setNewUserName] = useState("");
  const [newUserRole, setNewUserRole] = useState("photographer");
  const [creatingUser, setCreatingUser] = useState(false);
  const [createdUser, setCreatedUser] = useState<{ email: string; password: string; role: string } | null>(null);
  // --- Users Table State ---
  const [userList, setUserList] = useState<{ name: string; email: string; role: string; password: string }[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  // --- Matching State ---
  const [matching, setMatching] = useState(false);
  const [creatingEvent, setCreatingEvent] = useState(false);
  const [createdEvent, setCreatedEvent] = useState<any>(null);
  // --- Tabs State ---
  const [activeTab, setActiveTab] = useState("event");
  const BATCH_SIZE = 5;
  const CONCURRENCY = 3;

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreatingUser(true);
    setCreatedUser(null);
    try {
      const res = await createUserAsAdmin(newUserName, newUserRole);
      setCreatedUser(res);
      setNewUserName("");
      setNewUserRole("photographer");
    } catch (err: any) {
      alert(err.message || "Failed to create user");
    }
    setCreatingUser(false);
  };

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

  const fetchImages = async () => {
    try {
      const res = await getImages();
      setImageList(Array.isArray(res) ? res : res.images || []);
    } catch (err) {
      console.error("Failed to fetch images:", err);
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
      setUploading(true);
      setUploadProgress(0);

      const limit = pLimit(CONCURRENCY);

      try {
        const batches = [];

        for (let i = 0; i < selectedFiles.length; i += BATCH_SIZE) {
          const batch = selectedFiles.slice(i, i + BATCH_SIZE);
          const formData = new FormData();
          batch.forEach((file) => formData.append("files", file));
          batches.push(() =>
            uploadImages(formData).then((res) => {
              const completed = Math.min(i + BATCH_SIZE, selectedFiles.length);
              setUploadProgress(Math.round((completed / selectedFiles.length) * 100));
              return res;
            })
          );
        }

        await Promise.all(batches.map((fn) => limit(fn)));

        alert("All images uploaded successfully!");
        setSelectedFiles([]);
        if (isAdmin || isEditor) await fetchImages();
      } catch (err: any) {
        console.error(err);
        alert(`Upload failed: ${err.message}`);
      }

      setUploading(false);
      setUploadProgress(0);
    };




  // Remove handleCreateEvent and handleMatchFaces functions

  // --- Create New Event Handler (Admin Only) ---
  const handleCreateEvent = async () => {
    // First warning: Basic confirmation
    if (!confirm("Are you sure you want to create a new event?")) {
      return;
    }

    // Second warning: Data loss warning
    if (!confirm("⚠️ WARNING: Creating a new event may affect existing data and system performance. Continue?")) {
      return;
    }

    // Third warning: Resource usage warning
    if (!confirm("⚠️ SYSTEM WARNING: This action will allocate system resources and may take several minutes. Proceed?")) {
      return;
    }

    // Fourth warning: Final confirmation
    if (!confirm("⚠️ FINAL CONFIRMATION: You are about to create a new event. This action cannot be easily undone. Click OK to proceed.")) {
      return;
    }

    setCreatingEvent(true);
    setCreatedEvent(null);
    try {
      const res = await createNewEvent();
      setCreatedEvent(res);
      toast.success(res.message || "Event created successfully");
    } catch (err: any) {
      toast.error(err.message || "Failed to create event");
    }
    setCreatingEvent(false);
  };

  const fetchUserList = async () => {
    setLoadingUsers(true);
    try {
      const res = await getAllUsersForAdmin();
      setUserList(res.users || []);
    } catch (err) {
      setUserList([]);
    }
    setLoadingUsers(false);
  };

  const handleDeleteUser = async (email: string) => {
    if (!confirm(`Are you sure you want to delete user ${email}?`)) return;
    try {
      await deleteUserAsAdmin(email);
      await fetchUserList();
    } catch (err: any) {
      alert(err.message || "Failed to delete user");
    }
  };

  // --- Match Faces Handler (Admin Only) ---
  const handleMatchFaces = async () => {
    setMatching(true);
    try {
      const res = await matchFaces();
      toast.success(res.message || "Face matching started");
    } catch (err: any) {
      toast.error(err.message || "Failed to start face matching");
    }
    setMatching(false);
  };


  useEffect(() => {
    (async () => {
      const sessionUser = await getSession();
      if (sessionUser) {
        if (sessionUser.role !== "admin" && sessionUser.role !== "photographer" && sessionUser.role !== "editor") {
          router.push("/");
          return;
        }
        setUser(sessionUser);
        setChecking(false);
        // Only fetch user list if admin
        if (sessionUser.role === "admin") {
          await fetchUserList();
        }
        // Only fetch images if admin or editor (for download section)
        if (sessionUser.role === "admin" || sessionUser.role === "editor") {
          await fetchImages();
        }
      } else {
        router.push("/login");
      }
    })();
  }, [router]);

  // Remove useEffect that fetches user list on user change (handled above)

  if (checking) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p>Checking authentication…</p>
      </div>
    );
  }

  // --- Role-based UI ---
  const isAdmin = user?.role === "admin";
  const isPhotographer = user?.role === "photographer";
  const isEditor = user?.role === "editor";

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

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

      <h2 className="text-2xl font-semibold my-6 px-3 sm:px-10">Admin Panel</h2>

      <div className="px-3 sm:px-10">
        {(isAdmin || isEditor || isPhotographer) && (
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="mb-6">
              <TabsTrigger value="event">Event Management</TabsTrigger>
              {isAdmin && <TabsTrigger value="users">User Management</TabsTrigger>}
            </TabsList>
            <TabsContent value="event">
              {/* Event Management Tab: event creation, match faces, upload, images */}
              <div className="flex flex-row gap-2 mb-4">
                {isAdmin && (
                  <Button onClick={handleCreateEvent} disabled={creatingEvent}>
                    {creatingEvent ? "Creating Event…" : "Create New Event"}
                  </Button>
                )}
                {isAdmin && (
                  <Button onClick={handleMatchFaces} disabled={matching}>
                    {matching ? "Matching Faces…" : "Match Faces"}
                  </Button>
                )}
              </div>
              {isAdmin && createdEvent && (
                <div className=" border border-green-300 rounded p-3 mb-4">
                  <div className="font-medium">Event Created!</div>
                  {createdEvent.code && (
                    <div>Event Code: <span className="font-mono">{createdEvent.code}</span></div>
                  )}
                  {createdEvent.id && (
                    <div>Event ID: <span className="font-mono">{createdEvent.id}</span></div>
                  )}
                </div>
              )}
              {/* Upload Section: Photographers, Editors, Admin */}
              {(isAdmin || isPhotographer || isEditor) && (
                <div className="space-y-2 mb-6">
                  <label className="font-medium">Upload Images</label>
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleFileChange}
                    className="block"
                  />
                  <Button onClick={handleUpload} disabled={!selectedFiles.length || uploading}>
                    {uploading ? "Uploading…" : "Upload Selected Images"}
                  </Button>
                  {uploading && (
                    <div className="mt-2 w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                      <div
                        className="h-4 bg-blue-500 transition-all duration-300 ease-out"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  )}
                  <p className="text-sm mt-1 text-muted-foreground">
                    {uploading ? `Uploading... ${uploadProgress}%` : ""}
                  </p>
                </div>
              )}
              {/* Images grid (admin/editor) */}
              {(isAdmin || isEditor) && (
                <div className="space-y-2">
                  <h3 className="text-lg font-semibold">Uploaded Images</h3>
                  {imageList.length ? (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                      {imageList.map((img) => (
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
                  ) : (
                    <p className="text-muted-foreground text-sm">No images uploaded yet.</p>
                  )}
                </div>
              )}
            </TabsContent>
            {isAdmin && (
              <TabsContent value="users">
                {/* User Management Tab */}
                <div className="border rounded-lg p-4 mb-6 ">
                  <h3 className="text-lg font-semibold mb-2">Manage Users</h3>
                  <form className="flex flex-col sm:flex-row gap-2 items-start sm:items-end" onSubmit={handleCreateUser}>
                    <div>
                      <label className="block text-sm font-medium">Name</label>
                      <input
                        type="text"
                        value={newUserName}
                        onChange={e => setNewUserName(e.target.value)}
                        required
                        className="border rounded px-2 py-1 w-40"
                        placeholder="Enter name"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium">Role</label>
                      <select
                        value={newUserRole}
                        onChange={e => setNewUserRole(e.target.value)}
                        className="border rounded px-2 py-1 w-40"
                      >
                        <option value="photographer">Photographer</option>
                        <option value="editor">Editor</option>
                      </select>
                    </div>
                    <Button type="submit" disabled={creatingUser || !newUserName} className="mt-4 sm:mt-0">
                      {creatingUser ? "Creating..." : "Create User"}
                    </Button>
                  </form>
                  {createdUser && (
                    <div className="mt-4 border border-green-300 rounded p-3">
                      <div className="font-medium">User Created!</div>
                      <div>Email: <span className="font-mono">{createdUser.email}</span></div>
                      <div>Password: <span className="font-mono">{createdUser.password}</span></div>
                      <div>Role: <span className="font-mono">{createdUser.role}</span></div>
                    </div>
                  )}
                </div>
                <div className="border rounded-lg p-4 mb-6">
                  <h3 className="text-lg font-semibold mb-2">All Users (Photographers & Editors)</h3>
                  {loadingUsers ? (
                    <div>Loading users…</div>
                  ) : userList.length === 0 ? (
                    <div>No users found.</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="min-w-full border text-sm">
                        <thead>
                          <tr className="">
                            <th className="border px-2 py-1">Name</th>
                            <th className="border px-2 py-1">Email</th>
                            <th className="border px-2 py-1">Role</th>
                            <th className="border px-2 py-1">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {userList.map((u, idx) => (
                            <tr key={u.email + idx}>
                              <td className="border px-2 py-1 font-mono">{u.name}</td>
                              <td className="border px-2 py-1 font-mono">{u.email}</td>
                              <td className="border px-2 py-1">{u.role}</td>
                              <td className="border px-2 py-1">
                                <Button variant="destructive" size="sm" onClick={() => handleDeleteUser(u.email)}>
                                  Delete
                                </Button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <Button onClick={fetchUserList} className="mt-2">Refresh List</Button>
                </div>
              </TabsContent>
            )}
          </Tabs>
        )}
      </div>
    </div>
  );
}
