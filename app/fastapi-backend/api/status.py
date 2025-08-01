from fastapi import APIRouter
import os
import asyncio
from core.config import CDN_STORAGE_PATH

router = APIRouter()

def is_mounted(path: str) -> bool:
    return os.path.ismount(path)

@router.get("/cdn-mounted", tags=["status"])
async def check_cdn_mounted():
    mounted = await asyncio.to_thread(is_mounted, CDN_STORAGE_PATH)
    return {"mounted": mounted}
