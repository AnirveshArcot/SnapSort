import os
import subprocess
import threading
import asyncio
import time
from dotenv import load_dotenv
from services.faiss_index import load_faiss_index, save_faiss_index
from db.mongo import users_collection, feature_vector_collection, user_id_map, settings_coll
from services.model_loader import load_models, unload_models
load_dotenv()

ADMIN_MAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
ALGORITHM = "HS256"

CDN_STORAGE_PATH = os.getenv("CDN_STORAGE_PATH", "/home/ubuntu/SnapSort/app/fastapi-backend/cdn_storage")
REMOTE_USER = os.getenv("REMOTE_USER")
REMOTE_HOST = os.getenv("REMOTE_HOST")
REMOTE_PATH = os.getenv("REMOTE_PATH")

IMAGE_STORAGE_PATH = f"{CDN_STORAGE_PATH}/image_storage"
FAISS_INDEX_DIR = f"{CDN_STORAGE_PATH}/faiss_indices"
CACHE_DIR = "/home/ubuntu/SnapSort/app/fastapi-backend/image_cache"

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
SIMILARITY_THRESHOLD = 0.5
dimension = 128

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_current_event_id = None
_faiss_index = None


def is_mounted(path: str) -> bool:
    return os.path.ismount(path)


def mount_sshfs():
    try:
        print("Attempting to mount SSHFS with password...")
        subprocess.run(
            [
                "sshfs",
                f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}",
                CDN_STORAGE_PATH,
                "-o", "idmap=user",
                "-o", "IdentityFile=/home/ubuntu/.ssh/windows_key",
            ],
            check=True
        )
        if is_mounted(CDN_STORAGE_PATH):
            print("SSHFS mounted successfully.")
            os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        else:
            print("Mount point not detected after SSHFS attempt.")
    except subprocess.CalledProcessError as e:
        print(f"SSHFS mount failed: {e}")


def unmount_sshfs():
    try:
        print("Attempting to unmount SSHFS...")
        subprocess.run(["umount", CDN_STORAGE_PATH], check=True)
        print("Unmounted SSHFS successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to unmount SSHFS: {e}")


def start_mount_thread():
    def auto_mount_loop():
        while True:
            if not is_mounted(CDN_STORAGE_PATH):
                mount_sshfs()
            time.sleep(10)

    thread = threading.Thread(target=auto_mount_loop, daemon=True)
    thread.start()


async def get_current_event_id() -> str | None:
    global _current_event_id
    if _current_event_id is not None:
        return _current_event_id
    doc = await settings_coll.find_one({"_id": "current_event"})
    if doc:
        _current_event_id = str(doc["event_id"])
    return _current_event_id


async def set_current_event_id(event_id: str):
    global _current_event_id
    _current_event_id = event_id
    print(f"[DEBUG] Setting current_event_id to: {event_id}")
    await settings_coll.update_one(
        {"_id": "current_event"},
        {"$set": {"event_id": event_id}},
        upsert=True
    )


async def get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        current_event_id = await get_current_event_id()
        if current_event_id:
            _faiss_index = load_faiss_index(current_event_id, dimension)
    return _faiss_index


async def set_faiss_index(index):
    global _faiss_index
    _faiss_index = index
    current_event_id = await get_current_event_id()
    if current_event_id:
        save_faiss_index(index, current_event_id)

async def lifespan(app):
    print("Starting up application")

    current_event_id = await get_current_event_id()
    if current_event_id is None:
        await set_current_event_id("default_event")

    await get_faiss_index()
    await settings_coll.update_one(
        {"_id": "current_event"},
        {"$set": {"status": "free"}},
        upsert=True
    )
    start_mount_thread()
    yield
    print("Cleaning up application")
    unmount_sshfs()
