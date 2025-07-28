# api/event.py
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List
from db.models import UploadImagesRequest, UploadImagesResponse, Base64Image, UserOut
from core.config import ALLOWED_EXTENSIONS, IMAGE_STORAGE_PATH, CACHE_DIR, get_current_event_id
from services.image_processing import decode_base64_image, get_cached_compressed_image, cache_compressed_image
from api.auth import get_me as get_current_user
import os, base64, json
import cv2

router = APIRouter()

@router.post("/upload-images", response_model=UploadImagesResponse)
async def upload_images(req: UploadImagesRequest, current_user: UserOut = Depends(get_current_user)):
    if current_user.role not in ["photographer", "editor"]:
        raise HTTPException(status_code=403, detail="Not allowed to upload images")

    event_id = get_current_event_id()
    if not event_id:
        raise HTTPException(status_code=500, detail="Current event not set")

    if current_user.role == "editor":
        event_folder = os.path.join(IMAGE_STORAGE_PATH, f"{event_id}_edited")
    else:
        event_folder = os.path.join(IMAGE_STORAGE_PATH, event_id)

    os.makedirs(event_folder, exist_ok=True)
    uploaded = []

    for img in req.images:
        header, _, payload = img.base64.partition(",")
        image_data = base64.b64decode(payload or img.base64)
        base_name, ext = os.path.splitext(img.filename)
        original_filename = f"{base_name}_original{ext}"
        original_path = os.path.join(event_folder, original_filename)

        with open(original_path, "wb") as fout:
            fout.write(image_data)

        uploaded.append(original_filename)

    return UploadImagesResponse(uploaded=uploaded)


@router.get("/download")
async def download_image(filename: str = Query(...), current_user: UserOut = Depends(get_current_user)):
    event_id = get_current_event_id()
    if not event_id:
        raise HTTPException(status_code=500, detail="Current event not set")

    folder = os.path.join(IMAGE_STORAGE_PATH, event_id)

    if "_preview" in filename:
        base_name = filename.replace("_preview", "").rsplit(".", 1)[0]
    else:
        base_name = os.path.splitext(filename)[0]

    if current_user.role == "editor":
        for ext in ALLOWED_EXTENSIONS:
            original_filename = f"{base_name}_original{ext}"
            file_path = os.path.join(folder, original_filename)
            if os.path.exists(file_path):
                return FileResponse(path=file_path, filename=original_filename, media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail="File not found")
    else:
        compressed_filename = f"{base_name}_compressed.jpeg"
        cache_path = os.path.join(CACHE_DIR, compressed_filename)

        cached_img = get_cached_compressed_image(compressed_filename)
        if cached_img is not None and os.path.exists(cache_path):
            return FileResponse(path=cache_path, filename=compressed_filename, media_type="application/octet-stream")

        file_path = os.path.join(folder, compressed_filename)
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            cache_compressed_image(compressed_filename, img)
            return FileResponse(path=file_path, filename=compressed_filename, media_type="application/octet-stream")

        raise HTTPException(status_code=404, detail="File not found")


@router.get("/get-images", response_model=List[dict])
async def get_images(current_user: UserOut = Depends(get_current_user)):
    event_id = get_current_event_id()
    if not event_id:
        raise HTTPException(status_code=500, detail="Current event not set")

    event_folder = os.path.join(IMAGE_STORAGE_PATH, event_id)
    images = []

    if not os.path.exists(event_folder):
        return images

    if current_user.role in ["admin", "editor"]:
        for f in os.listdir(event_folder):
            if f.lower().endswith("_preview.jpeg"):
                file_path = os.path.join(event_folder, f)
                try:
                    with open(file_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        images.append({
                            "name": f,
                            "base64": f"data:image/jpeg;base64,{encoded_string}"
                        })
                except Exception:
                    continue
        return images

    if current_user.role == "photographer":
        return []

    matches_path = os.path.join(event_folder, "matches.json")
    if not os.path.exists(matches_path):
        return images

    try:
        with open(matches_path, "r") as f:
            matches_data = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid matches.json format")

    matched_files = matches_data.get("matches", {}).get(current_user.id, [])

    for idx, filename in enumerate(matched_files):
        base, ext = os.path.splitext(filename)
        base = base.replace("_compressed", "")
        preview_filename = f"{base}_preview.jpeg"
        preview_path = os.path.join(event_folder, preview_filename)

        if not os.path.exists(preview_path):
            continue

        try:
            with open(preview_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                images.append({
                    "name": preview_filename,
                    "base64": f"data:image/jpeg;base64,{encoded_string}"
                })
        except Exception:
            continue

    return images
