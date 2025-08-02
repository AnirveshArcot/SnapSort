import { NextResponse } from "next/server";

const API_URL = "/api";

async function fetchAPI(endpoint: string, options: RequestInit = {}) {
  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    credentials: 'include',
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || 'An error occurred')
  return data
}

async function fetchAPIMedia(endpoint: string, options: RequestInit = {}) {
  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    credentials: 'include',
  })
  if (!res.ok) throw new Error('An error occurred during media')
  return res
}

export async function registerUser(userData: Record<string, any>) {
  return fetchAPI('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(userData),
  })
}



export async function logoutUser() {
  return fetchAPI('/auth/logout', { method: 'POST' })
}





export async function loginUser(email: string, password: string) {
  const formData = new URLSearchParams();
  formData.append("username", email);
  formData.append("password", password);

  const isAdmin = email.toLowerCase().endsWith("arka.ai");
  const endpoint = isAdmin ? "/auth/login" : "/auth/login";

  return fetchAPI(endpoint, {
    method: "POST",
    body: formData,
    credentials: "include",
  });
}


export async function getSession() {
  try {
    return await fetchAPI('/auth/me')
  } catch (err) {
    console.error('Error fetching session:', err)
    return null
  }
}


export async function uploadImages(formData: FormData) {
  return await fetchAPI("/event/upload-images", {
    method: "POST",
    body: formData,
  });
}


export async function getImages(skip: number = 0, limit: number = 20) {
  return await fetchAPI(`/event/get-images?skip=${skip}&limit=${limit}`);
}


export async function downloadImageBlob(filename: string): Promise<Blob> {
  const res = await fetchAPIMedia(`/event/download?filename=${encodeURIComponent(filename)}`);
  return await res.blob();
}

export async function checkCDNMounted(){
  return await fetchAPI('/admin/check-cdn');
}


export async function createUserAsAdmin(name: string, role: string) {
  return fetchAPI('/admin/create-user', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, role }),
  });
}

export async function createNewEvent() {
  return await fetchAPI("/event/create-event", {
    method: "POST",
  });
}


export async function getAllUsersForAdmin() {
  return fetchAPI('/admin/list-users', {
    method: 'GET',
  });
}

export async function deleteUserAsAdmin(email: string) {
  return fetchAPI('/admin/delete-user', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
}

export async function matchFaces() {
  return await fetchAPI("/event/match-faces", {
    method: "POST",
  });
}