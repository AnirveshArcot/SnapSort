# api/status.py
from fastapi import APIRouter
import os
from core.config import CDN_STORAGE_PATH

router = APIRouter()

def is_mounted(path: str) -> bool:
    return os.path.ismount(path)

@router.get("/cdn-mounted", tags=["status"])
def check_cdn_mounted():
    mounted = is_mounted(CDN_STORAGE_PATH)
    return {"mounted": mounted}
