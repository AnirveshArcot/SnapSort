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
  const endpoint = isAdmin ? "/auth/admin-login" : "/auth/login";

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


export async function getEventImages(){
  try {
    return await fetchAPI('/event/get-images');
  } catch (err) {
    console.error('Error fetching event images:', err);
    return null;
  }
}


export async function uploadEventImages(images: { filename: string, base64: string }[]) {
  return await fetchAPI("/event/upload-images", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images }),
  });
}


export async function createNewEvent() {
  return await fetchAPI("/event/create-event", {
    method: "POST",
  });
}

export async function matchFaces() {
  return await fetchAPI("/event/match-faces", {
    method: "POST",
  });
}



export async function getPreviwes() {
  const res = await fetchAPI("/event/previews");
  return res.images; 
};

export async function downloadImageBlob(filename: string): Promise<Blob> {
  const res = await fetchAPIMedia(`/event/download?filename=${encodeURIComponent(filename)}`);
  return await res.blob(); 
}