"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import { UserNav } from "@/components/user-nav";
import { Button } from "@/components/ui/button";
import {
  uploadImages,
  getImages,
  downloadImageBlob,
  createUserAsAdmin,
  getAllUsersForAdmin,
  deleteUserAsAdmin,
  getSession,
  matchFaces,
  createNewEvent,
} from "@/lib/api";
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
  const [uploadProgress, setUploadProgress] = useState(0);

  const [imageList, setImageList] = useState<{ name: string; base64: string }[]>([]);
  const [page, setPage] = useState(0);
  const [totalPages, setTotalPages] = useState(1);

  const [newUserName, setNewUserName] = useState("");
  const [newUserRole, setNewUserRole] = useState("photographer");
  const [creatingUser, setCreatingUser] = useState(false);
  const [createdUser, setCreatedUser] = useState<{ email: string; password: string; role: string } | null>(null);
  const [userList, setUserList] = useState<{ name: string; email: string; role: string; password: string }[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);

  const [matching, setMatching] = useState(false);
  const [creatingEvent, setCreatingEvent] = useState(false);
  const [createdEvent, setCreatedEvent] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("event");

  const BATCH_SIZE = 5;
  const CONCURRENCY = 3;

  const isAdmin = user?.role === "admin";
  const isPhotographer = user?.role === "photographer";
  const isEditor = user?.role === "editor";

  const fetchImages = async (targetPage: number = 0) => {
    try {
      const res = await getImages(targetPage * 20, 20);
      const images = Array.isArray(res) ? res : res.images || [];

      setImageList(images);
      setPage(targetPage);
      const count = res.total_count || 0;
      setTotalPages(Math.max(1, Math.ceil(count / 20)));
    } catch (err) {
      console.error("Failed to fetch images:", err);
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
      console.error("Download failed:", err);
      alert("Failed to download image.");
    }
  };

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
      toast.success("Images uploaded!");
      setSelectedFiles([]);
      await fetchImages(page);
    } catch (err: any) {
      console.error(err);
      toast.error(`Upload failed: ${err.message}`);
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
    if (!confirm(`Delete user ${email}?`)) return;
    try {
      await deleteUserAsAdmin(email);
      await fetchUserList();
    } catch (err: any) {
      alert(err.message || "Failed to delete user");
    }
  };

  const handleCreateEvent = async () => {
    if (!confirm("Are you sure you want to create a new event?")) return;
    if (!confirm("⚠️ Creating a new event may affect existing data. Continue?")) return;
    if (!confirm("⚠️ This may take several minutes. Proceed?")) return;
    if (!confirm("⚠️ FINAL CONFIRMATION: Proceed with event creation?")) return;

    setCreatingEvent(true);
    try {
      const res = await createNewEvent();
      setCreatedEvent(res);
      toast.success(res.message || "Event created");
    } catch (err: any) {
      toast.error(err.message || "Failed to create event");
    }
    setCreatingEvent(false);
  };

  const handleMatchFaces = async () => {
    setMatching(true);
    try {
      const res = await matchFaces();
      toast.success(res.message || "Matching started");
    } catch (err: any) {
      toast.error(err.message || "Failed to match faces");
    }
    setMatching(false);
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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

  useEffect(() => {
    (async () => {
      const sessionUser = await getSession();
      if (sessionUser) {
        if (!["admin", "photographer", "editor"].includes(sessionUser.role)) {
          router.push("/");
          return;
        }
        setUser(sessionUser);
        setChecking(false);
        if (sessionUser.role === "admin") await fetchUserList();
        if (["admin", "editor"].includes(sessionUser.role)) await fetchImages(0);
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
    <div>
      {/* Header */}
      <header className="border-b">
        <div className="flex h-16 items-center justify-between px-3 sm:px-10">
          <Link href="/" className="flex items-center">
            <span className="text-xl font-bold">SnapSort</span>
          </Link>
          <nav className="flex items-center">
            {user ? (
              <UserNav user={user} />
            ) : (
              <div className="flex gap-2">
                <Link href="/login"><Button variant="ghost">Login</Button></Link>
                <Link href="/register"><Button>Register</Button></Link>
              </div>
            )}
          </nav>
        </div>
      </header>

      <h2 className="text-2xl font-semibold my-6 px-3 sm:px-10">Admin Panel</h2>

      <div className="px-3 sm:px-10">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-6">
            <TabsTrigger value="event">Event Management</TabsTrigger>
            {isAdmin && <TabsTrigger value="users">User Management</TabsTrigger>}
          </TabsList>

          {/* Event Management */}
          <TabsContent value="event">
            {(isAdmin || isPhotographer || isEditor) && (
              <>
                <div className="flex flex-wrap gap-2 mb-4">
                  {isAdmin && (
                    <Button onClick={handleCreateEvent} disabled={creatingEvent}>
                      {creatingEvent ? "Creating…" : "Create New Event"}
                    </Button>
                  )}
                  {isAdmin && (
                    <Button onClick={handleMatchFaces} disabled={matching}>
                      {matching ? "Matching…" : "Match Faces"}
                    </Button>
                  )}
                </div>
                {createdEvent && (
                  <div className="border border-green-300 rounded p-3 mb-4">
                    <div className="font-medium">Event Created!</div>
                    <div>Code: <span className="font-mono">{createdEvent.code}</span></div>
                    <div>ID: <span className="font-mono">{createdEvent.id}</span></div>
                  </div>
                )}

                {/* Upload */}
                <div className="space-y-2 mb-6">
                  <label className="font-medium">Upload Images</label>
                  <input type="file" multiple accept="image/*" onChange={handleFileChange} />
                  <Button onClick={handleUpload} disabled={!selectedFiles.length || uploading}>
                    {uploading ? "Uploading…" : "Upload"}
                  </Button>
                  {uploading && (
                    <div className="bg-gray-200 h-4 rounded-full overflow-hidden mt-1">
                      <div
                        className="h-4 bg-blue-500"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  )}
                </div>

                {/* Gallery */}
                <div className="space-y-2">
                  <h3 className="text-lg font-semibold">Uploaded Images</h3>
                  {imageList.length ? (
                    <>
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                        {imageList.map((img) => (
                          <div key={img.name} className="border rounded-lg shadow">
                            <img src={img.base64} alt={img.name} className="w-full h-48 object-cover" />
                            <div className="p-2 flex justify-between items-center text-sm">
                              <span className="truncate" title={img.name}>{img.name}</span>
                              <Button size="sm" onClick={() => handleDownload(img.name)}>Download</Button>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="mt-4 flex flex-col sm:flex-row gap-2 sm:gap-4 items-center justify-between">
                        <div className="flex gap-2">
                          <Button onClick={() => fetchImages(page - 1)} disabled={page === 0}>
                            Previous
                          </Button>
                          <Button onClick={() => fetchImages(page + 1)} disabled={page + 1 >= totalPages}>
                            Next
                          </Button>
                        </div>

                        <div className="flex items-center gap-2">
                          <span>
                            Page <strong>{page + 1}</strong> of <strong>{totalPages}</strong>
                          </span>
                          <input
                            type="number"
                            min={1}
                            max={totalPages}
                            defaultValue={page + 1}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                const target = e.target as HTMLInputElement;
                                const pageNumber = parseInt(target.value);
                                if (!isNaN(pageNumber) && pageNumber >= 1 && pageNumber <= totalPages) {
                                  fetchImages(pageNumber - 1);
                                }
                              }
                            }}
                            className="w-16 px-2 py-1 border rounded text-center"
                          />
                          <span className="text-sm text-muted-foreground">Press Enter to jump</span>
                        </div>
                      </div>

                    </>
                  ) : (
                    <p className="text-muted-foreground text-sm">No images uploaded yet.</p>
                  )}
                </div>
              </>
            )}
          </TabsContent>

          {/* User Management */}
          {isAdmin && (
            <TabsContent value="users">
              <div className="border rounded-lg p-4 mb-6">
                <h3 className="text-lg font-semibold mb-2">Manage Users</h3>
                <form onSubmit={handleCreateUser} className="flex flex-col sm:flex-row gap-2">
                  <input type="text" value={newUserName} onChange={e => setNewUserName(e.target.value)} required placeholder="Name" className="border px-2 py-1 rounded" />
                  <select value={newUserRole} onChange={e => setNewUserRole(e.target.value)} className="border px-2 py-1 rounded">
                    <option value="photographer">Photographer</option>
                    <option value="editor">Editor</option>
                  </select>
                  <Button type="submit" disabled={creatingUser}>Create</Button>
                </form>
                {createdUser && (
                  <div className="mt-4 border border-green-300 rounded p-3">
                    <div className="font-medium">User Created!</div>
                    <div>Email: {createdUser.email}</div>
                    <div>Password: {createdUser.password}</div>
                    <div>Role: {createdUser.role}</div>
                  </div>
                )}
              </div>
              <div className="border rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-2">All Users</h3>
                {loadingUsers ? (
                  <p>Loading users…</p>
                ) : (
                  <table className="w-full border text-sm">
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
                )}
              </div>
            </TabsContent>
          )}
        </Tabs>
      </div>
    </div>
  );
}
