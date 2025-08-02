from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from typing import List
from grpc import Status
from db.models import UploadImagesRequest, UploadImagesResponse, Base64Image, UserOut
from core.config import ALLOWED_EXTENSIONS, IMAGE_STORAGE_PATH, CACHE_DIR, get_current_event_id
from services.image_processing import decode_base64_image, get_cached_compressed_image, cache_compressed_image
from api.auth import get_me as get_current_user
import os
import base64
import json
import aiofiles
import cv2
import numpy as np

router = APIRouter()

@router.post("/upload-images", response_model=UploadImagesResponse)
async def upload_images(
    files: List[UploadFile] = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    if current_user.role not in ["photographer", "editor", "admin"]:
        raise HTTPException(status_code=403, detail="Not allowed to upload images")

    event_id = await get_current_event_id()
    if not event_id:
        raise HTTPException(status_code=500, detail="Current event not set")

    event_folder = os.path.join(IMAGE_STORAGE_PATH, f"{event_id}_edited" if current_user.role == "editor" else event_id)
    os.makedirs(event_folder, exist_ok=True)
    uploaded = []

    for file in files:
        contents = await file.read()
        filename = file.filename
        base_name, ext = os.path.splitext(filename)

        # Editor gets a single edited version
        if current_user.role == "editor":
            filename = f"{base_name}_edited{ext}"
            file_path = os.path.join(event_folder, filename)
            async with aiofiles.open(file_path, "wb") as fout:
                await fout.write(contents)
            uploaded.append(filename)
            continue

        # Photographer: Save original
        original_filename = f"{base_name}_original{ext}"
        original_path = os.path.join(event_folder, original_filename)
        async with aiofiles.open(original_path, "wb") as fout:
            await fout.write(contents)
        uploaded.append(original_filename)

        try:
            nparr = np.frombuffer(contents, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None:
                raise ValueError("Decoded image is None")

            # Preview
            preview = cv2.resize(img_np, (0, 0), fx=0.25, fy=0.25)
            preview_filename = f"{base_name}_preview.jpeg"
            preview_path = os.path.join(event_folder, preview_filename)
            quality = 60
            while True:
                success, buffer = cv2.imencode(".jpeg", preview, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    raise ValueError("Failed to encode preview image")
                if len(buffer) <= 50 * 1024 or quality <= 30:
                    break
                quality -= 5
            async with aiofiles.open(preview_path, "wb") as fout:
                await fout.write(buffer.tobytes())
            uploaded.append(preview_filename)

            # Compressed
            compressed_filename = f"{base_name}_compressed.jpeg"
            compressed_path = os.path.join(event_folder, compressed_filename)
            quality = 80
            while True:
                success, buffer = cv2.imencode(".jpeg", img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    raise ValueError("Failed to encode compressed image")
                if len(buffer) <= 512 * 1024 or quality <= 40:
                    break
                quality -= 5
            async with aiofiles.open(compressed_path, "wb") as fout:
                await fout.write(buffer.tobytes())
            uploaded.append(compressed_filename)

        except Exception as e:
            raise HTTPException(status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process image {file.filename}: {e}")

    return UploadImagesResponse(uploaded=uploaded)



@router.get("/download")
async def download_image(filename: str = Query(...), current_user: UserOut = Depends(get_current_user)):
    event_id = await get_current_event_id()
    if not event_id:
        raise HTTPException(status_code=500, detail="Current event not set")

    folder = os.path.join(IMAGE_STORAGE_PATH, event_id)
    base_name = filename.replace("_preview", "").rsplit(".", 1)[0] if "_preview" in filename else os.path.splitext(filename)[0]

    if current_user.role == "editor":
        for ext in ALLOWED_EXTENSIONS:
            file_path = os.path.join(folder, f"{base_name}_original{ext}")
            if os.path.exists(file_path):
                return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail="File not found")

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
    event_id = await get_current_event_id()
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
                    async with aiofiles.open(file_path, "rb") as image_file:
                        encoded_string = base64.b64encode(await image_file.read()).decode("utf-8")
                        images.append({"name": f, "base64": f"data:image/jpeg;base64,{encoded_string}"})
                except Exception:
                    continue
        return images

    if current_user.role == "photographer":
        return []

    matches_path = os.path.join(event_folder, "matches.json")
    if not os.path.exists(matches_path):
        return images

    try:
        async with aiofiles.open(matches_path, "r") as f:
            content = await f.read()
            matches_data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid matches.json format")

    matched_files = matches_data.get("matches", {}).get(current_user.id, [])

    for filename in matched_files:
        base = os.path.splitext(filename.replace("_compressed", ""))[0]
        preview_filename = f"{base}_preview.jpeg"
        preview_path = os.path.join(event_folder, preview_filename)

        if not os.path.exists(preview_path):
            continue

        try:
            async with aiofiles.open(preview_path, "rb") as image_file:
                encoded_string = base64.b64encode(await image_file.read()).decode("utf-8")
                images.append({"name": preview_filename, "base64": f"data:image/jpeg;base64,{encoded_string}"})
        except Exception:
            continue

    return images
