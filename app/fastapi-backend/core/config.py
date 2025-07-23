# core/config.py
import os
from dotenv import load_dotenv
from services.faiss_index import load_faiss_index
from db.mongo import users_collection, feature_vector_collection, user_id_map, settings_coll

load_dotenv()

ADMIN_MAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
ALGORITHM = "HS256"

CDN_STORAGE_PATH = "./cdn_storage/"
FAISS_INDEX_DIR = "./faiss_indices/"
CACHE_DIR = "./image_cache/"
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

SIMILARITY_THRESHOLD=0.5

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs(CDN_STORAGE_PATH, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

CURRENT_EVENT_ID = None
faiss_index = None
dimension = 4096

async def lifespan(app):
    global CURRENT_EVENT_ID, faiss_index

    doc = settings_coll.find_one({"_id": "current_event"})
    if doc:
        CURRENT_EVENT_ID = str(doc["event_id"])
    else:
        settings_coll.insert_one({"_id": "current_event", "event_id": CURRENT_EVENT_ID})

    faiss_index = load_faiss_index(CURRENT_EVENT_ID, dimension)
    settings_coll.update_one(
        {"_id": "current_event"}, {"$set": {"status": "free"}}, upsert=True
    )
    yield
