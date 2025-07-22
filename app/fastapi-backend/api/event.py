# api/event.py
from fastapi import APIRouter, Depends, HTTPException
from db.models import UploadImagesRequest, UploadImagesResponse, Base64Image, UserOut
from core.config import CDN_STORAGE_PATH
from services.image_processing import decode_base64_image
import os, base64

router = APIRouter()

@router.post("/upload-images", response_model=UploadImagesResponse)
def upload_images(req: UploadImagesRequest):
    uploaded = []
    for img in req.images:
        header, _, payload = img.base64.partition(",")
        image_data = base64.b64decode(payload or img.base64)
        file_path = os.path.join(CDN_STORAGE_PATH, img.filename)
        with open(file_path, "wb") as f:
            f.write(image_data)
        uploaded.append(img.filename)
    return UploadImagesResponse(uploaded=uploaded)

@router.get("/download")
def download_image(filename: str):
    path = os.path.join(CDN_STORAGE_PATH, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    from fastapi.responses import FileResponse
    return FileResponse(path, filename=filename)

@router.get("/get-images")
def get_images():
    files = os.listdir(CDN_STORAGE_PATH)
    return [{"name": f} for f in files if f.endswith(".jpeg") or f.endswith(".jpg")]