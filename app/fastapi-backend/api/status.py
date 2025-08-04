from fastapi import APIRouter
import asyncio
from core.config import is_mounted, mount_sshfs, CDN_STORAGE_PATH  # adjust import if needed

router = APIRouter()

@router.get("/cdn-mounted", tags=["status"])
async def check_cdn_mounted():
    mounted = await asyncio.to_thread(is_mounted, CDN_STORAGE_PATH)

    if mounted:
        return {"mounted": True}
    await asyncio.to_thread(mount_sshfs)
    for _ in range(2):
        mounted = await asyncio.to_thread(is_mounted, CDN_STORAGE_PATH)
        if mounted:
            return {"mounted": True}
        await asyncio.sleep(1)

    return {"mounted": False}
