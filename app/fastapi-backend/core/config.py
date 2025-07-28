import os
import subprocess
import threading
import time
from dotenv import load_dotenv
from services.faiss_index import load_faiss_index, save_faiss_index
from db.mongo import users_collection, feature_vector_collection, user_id_map, settings_coll

load_dotenv()

ADMIN_MAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
ALGORITHM = "HS256"

CDN_STORAGE_PATH="./cdn_storage/"
IMAGE_STORAGE_PATH = "./cdn_storage/image_storage/"
FAISS_INDEX_DIR = "./cdn_storage/faiss_indices/"
CACHE_DIR = "./image_cache/"
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
SIMILARITY_THRESHOLD = 0.5
dimension = 4096

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_current_event_id = None
_faiss_index = None


def is_mounted(path: str) -> bool:
    return os.path.ismount(path)


def mount_ftp():
    try:
        print("Attempting to mount FTP...")
        subprocess.run(
            ["sudo", "curlftpfs", "ftpuser:arka6969@122.166.210.200", CDN_STORAGE_PATH, "-o", "allow_other"],
            check=True
        )
        if is_mounted(CDN_STORAGE_PATH):
            print("FTP mounted successfully.")
            os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        else:
            print("Mount point not detected after mount attempt.")
    except subprocess.CalledProcessError as e:
        print(f"Mount attempt failed: {e}")



def unmount_ftp():
    try:
        print("Unmounting FTP...")
        subprocess.run(["sudo", "umount", "-l",CDN_STORAGE_PATH ], check=True)
        print("FTP unmounted.")
    except subprocess.CalledProcessError as e:
        print(f"Unmount failed: {e}")


def auto_mount_loop():
    while True:
        if not is_mounted(CDN_STORAGE_PATH):
            mount_ftp()
        time.sleep(10)


def start_mount_thread():
    thread = threading.Thread(target=auto_mount_loop, daemon=True)
    thread.start()


def get_current_event_id() -> str | None:
    global _current_event_id
    if _current_event_id is not None:
        return _current_event_id
    doc = settings_coll.find_one({"_id": "current_event"})
    if doc:
        _current_event_id = str(doc["event_id"])
    return _current_event_id


def set_current_event_id(event_id: str):
    global _current_event_id
    _current_event_id = event_id
    settings_coll.update_one({"_id": "current_event"}, {"$set": {"event_id": event_id}}, upsert=True)


def get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        current_event_id = get_current_event_id()
        if current_event_id:
            _faiss_index = load_faiss_index(current_event_id, dimension)
    return _faiss_index


def set_faiss_index(index):
    global _faiss_index
    _faiss_index = index
    current_event_id = get_current_event_id()
    if current_event_id:
        save_faiss_index(index, current_event_id)



async def lifespan(app):
    print("Starting up application")

    if get_current_event_id() is None:
        set_current_event_id("default_event")

    get_faiss_index()
    settings_coll.update_one({"_id": "current_event"}, {"$set": {"status": "free"}}, upsert=True)
    start_mount_thread()

    yield

    print("Cleaning up application")
    unmount_ftp()
