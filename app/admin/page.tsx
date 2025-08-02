"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { UserNav } from "@/components/user-nav";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { toast } from "sonner";
import pLimit from "p-limit";

import {
  uploadImages,
  getImages,
  downloadImageBlob,
  createUserAsAdmin,
  getAllUsersForAdmin,
  deleteUserAsAdmin,
  getSession,
  matchFaces,
  createNewEvent
} from "@/lib/api";

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
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [fetching, setFetching] = useState(false);

  const [newUserName, setNewUserName] = useState("");
  const [newUserRole, setNewUserRole] = useState("photographer");
  const [creatingUser, setCreatingUser] = useState(false);
  const [createdUser, setCreatedUser] = useState<{ email: string; password: string; role: string } | null>(null);

  const [userList, setUserList] = useState<{ name: string; email: string; role: string; password: string }[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);

  const [matching, setMatching] = useState(false);
  const [creatingEvent, setCreatingEvent] = useState(false);
  const [createdEvent, setCreatedEvent] = useState<any>(null);

  const BATCH_SIZE = 5;
  const CONCURRENCY = 3;

  const isAdmin = user?.role === "admin";
  const isPhotographer = user?.role === "photographer";
  const isEditor = user?.role === "editor";

  const fetchImages = useCallback(async (reset = false) => {
    if (fetching || (!reset && !hasMore)) return;
    setFetching(true);
    try {
      const currentPage = reset ? 0 : page;
      const newImages = await getImages(currentPage * 20, 20);
      if (reset) {
        setImageList(newImages);
        setPage(1);
      } else {
        setImageList((prev) => [...prev, ...newImages]);
        setPage((prev) => prev + 1);
      }
      setHasMore(newImages.length === 20);
    } catch (err) {
      console.error("Failed to fetch images:", err);
    } finally {
      setFetching(false);
    }
  }, [fetching, hasMore, page]);

  useEffect(() => {
    const handleScroll = () => {
      if (
        window.innerHeight + document.documentElement.scrollTop >=
          document.documentElement.offsetHeight - 100 &&
        hasMore &&
        !fetching
      ) {
        fetchImages();
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [fetchImages, hasMore, fetching]);

  useEffect(() => {
    (async () => {
      const sessionUser = await getSession();
      if (sessionUser) {
        if (!["admin", "editor", "photographer"].includes(sessionUser.role)) {
          router.push("/");
          return;
        }
        setUser(sessionUser);
        setChecking(false);
        if (sessionUser.role === "admin") await fetchUserList();
        if (["admin", "editor"].includes(sessionUser.role)) await fetchImages(true);
      } else {
        router.push("/login");
      }
    })();
  }, [router, fetchImages]);

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
          uploadImages(formData).then(() => {
            const completed = Math.min(i + BATCH_SIZE, selectedFiles.length);
            setUploadProgress(Math.round((completed / selectedFiles.length) * 100));
          })
        );
      }
      await Promise.all(batches.map((fn) => limit(fn)));
      alert("All images uploaded successfully!");
      setSelectedFiles([]);
      if (isAdmin || isEditor) await fetchImages(true);
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    }
    setUploading(false);
    setUploadProgress(0);
  };

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

  const handleDeleteUser = async (email: string) => {
    if (!confirm(`Are you sure you want to delete user ${email}?`)) return;
    try {
      await deleteUserAsAdmin(email);
      await fetchUserList();
    } catch (err: any) {
      alert(err.message || "Failed to delete user");
    }
  };

  const fetchUserList = async () => {
    setLoadingUsers(true);
    try {
      const res = await getAllUsersForAdmin();
      setUserList(res.users || []);
    } catch {
      setUserList([]);
    }
    setLoadingUsers(false);
  };

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

  const handleCreateEvent = async () => {
    if (!confirm("Are you sure you want to create a new event?")) return;
    if (!confirm("⚠️ WARNING: Creating a new event may affect existing data. Continue?")) return;
    if (!confirm("⚠️ SYSTEM WARNING: This may take time and use system resources. Proceed?")) return;
    if (!confirm("⚠️ FINAL CONFIRMATION: This action cannot be undone. Click OK to proceed.")) return;
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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

  const handleDownload = async (filename: string) => {
    try {
      const blob = await downloadImageBlob(filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename.replace("_preview", "");
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert("Failed to download image.");
    }
  };

  if (checking) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p>Checking authentication…</p>
      </div>
    );
  }

  return (
    <div>
      <header className="border-b">
        <div className="flex h-16 items-center justify-between px-3 sm:px-10">
          <Link href="/" className="flex items-center">
            <span className="text-xl font-bold">SnapSort</span>
          </Link>
          <nav className="flex items-center">
            {user ? <UserNav user={user} /> : (
              <div className="flex items-center gap-2">
                <Link href="/login"><Button variant="ghost">Login</Button></Link>
                <Link href="/register"><Button>Register</Button></Link>
              </div>
            )}
          </nav>
        </div>
      </header>

      <h2 className="text-2xl font-semibold my-6 px-3 sm:px-10">Admin Panel</h2>
      <div className="px-3 sm:px-10">
        <Tabs defaultValue="event" className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="event">Event Management</TabsTrigger>
            {isAdmin && <TabsTrigger value="users">User Management</TabsTrigger>}
          </TabsList>

          <TabsContent value="event">
            <div className="flex flex-row gap-2 mb-4">
              {isAdmin && <Button onClick={handleCreateEvent} disabled={creatingEvent}>{creatingEvent ? "Creating Event…" : "Create New Event"}</Button>}
              {isAdmin && <Button onClick={handleMatchFaces} disabled={matching}>{matching ? "Matching Faces…" : "Match Faces"}</Button>}
            </div>
            {isAdmin && createdEvent && (
              <div className="border border-green-300 rounded p-3 mb-4">
                <div className="font-medium">Event Created!</div>
                {createdEvent.code && <div>Event Code: <span className="font-mono">{createdEvent.code}</span></div>}
                {createdEvent.id && <div>Event ID: <span className="font-mono">{createdEvent.id}</span></div>}
              </div>
            )}
            {(isAdmin || isPhotographer || isEditor) && (
              <div className="space-y-2 mb-6">
                <label className="font-medium">Upload Images</label>
                <input type="file" multiple accept="image/*" onChange={handleFileChange} className="block" />
                <Button onClick={handleUpload} disabled={!selectedFiles.length || uploading}>
                  {uploading ? "Uploading…" : "Upload Selected Images"}
                </Button>
                {uploading && (
                  <div className="mt-2 w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                    <div className="h-4 bg-blue-500 transition-all duration-300 ease-out" style={{ width: `${uploadProgress}%` }} />
                  </div>
                )}
              </div>
            )}
            {(isAdmin || isEditor) && (
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">Uploaded Images</h3>
                {imageList.length ? (
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                    {imageList.map((img) => (
                      <div key={img.name} className="border rounded-lg overflow-hidden shadow hover:shadow-md transition">
                        <img src={img.base64} alt={img.name} className="w-full h-48 object-cover" />
                        <div className="p-2 flex justify-between items-center text-sm">
                          <span className="truncate" title={img.name}>{img.name}</span>
                          <Button onClick={() => handleDownload(img.name)}>Download</Button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-sm">No images uploaded yet.</p>
                )}
                {hasMore && <div className="text-sm text-center text-muted-foreground py-4">Loading more...</div>}
                {!hasMore && imageList.length > 0 && <div className="text-sm text-center text-muted-foreground py-4">All images loaded.</div>}
              </div>
            )}
          </TabsContent>

          {isAdmin && (
            <TabsContent value="users">
              <div className="border rounded-lg p-4 mb-6">
                <h3 className="text-lg font-semibold mb-2">Manage Users</h3>
                <form className="flex flex-col sm:flex-row gap-2 items-start sm:items-end" onSubmit={handleCreateUser}>
                  <div>
                    <label className="block text-sm font-medium">Name</label>
                    <input type="text" value={newUserName} onChange={e => setNewUserName(e.target.value)} required className="border rounded px-2 py-1 w-40" placeholder="Enter name" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium">Role</label>
                    <select value={newUserRole} onChange={e => setNewUserRole(e.target.value)} className="border rounded px-2 py-1 w-40">
                      <option value="photographer">Photographer</option>
                      <option value="editor">Editor</option>
                    </select>
                  </div>
                  <Button type="submit" disabled={creatingUser || !newUserName} className="mt-4 sm:mt-0">{creatingUser ? "Creating..." : "Create User"}</Button>
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
                <h3 className="text-lg font-semibold mb-2">All Users</h3>
                {loadingUsers ? (
                  <div>Loading users…</div>
                ) : userList.length === 0 ? (
                  <div>No users found.</div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="min-w-full border text-sm">
                      <thead>
                        <tr>
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
                              <Button variant="destructive" size="sm" onClick={() => handleDeleteUser(u.email)}>Delete</Button>
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
      </div>
    </div>
  );
}
