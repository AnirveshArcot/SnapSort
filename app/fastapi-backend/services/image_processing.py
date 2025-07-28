import base64
import re
import cv2
import numpy as np
import os
from fastapi import HTTPException
from core.config import ALLOWED_EXTENSIONS, IMAGE_STORAGE_PATH, CACHE_DIR
import json

def decode_base64_image(base64_string):
    base64_data = re.sub(r"^data:image/\w+;base64,", "", base64_string)
    image_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def fetch_image_from_cdn(image_name: str):
    path = os.path.join(IMAGE_STORAGE_PATH, image_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return cv2.imread(path)

# --- MFU Cache Functions (Single Event) ---
def get_free_space_gb(path):
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)

def _get_meta_path():
    return os.path.join(CACHE_DIR, "cache_meta.json")

def _load_meta():
    meta_path = _get_meta_path()
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}

def _save_meta(meta):
    meta_path = _get_meta_path()
    with open(meta_path, "w") as f:
        json.dump(meta, f)

def increment_download_count(filename):
    meta = _load_meta()
    meta[filename] = meta.get(filename, 0) + 1
    _save_meta(meta)

def evict_least_frequently_used():
    meta = _load_meta()
    if not meta:
        return
    lfu_file = min(meta, key=meta.get)
    lfu_path = os.path.join(CACHE_DIR, lfu_file)
    try:
        os.remove(lfu_path)
    except Exception:
        pass
    meta.pop(lfu_file, None)
    _save_meta(meta)

def get_cached_compressed_image(filename):
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        increment_download_count(filename)
        return cv2.imread(cache_path)
    return None

def cache_compressed_image(filename, image_data):
    cache_path = os.path.join(CACHE_DIR, filename)
    # Only cache if at least 5GB free, otherwise evict MFU
    while not os.path.exists(cache_path) and get_free_space_gb(CACHE_DIR) < 5.0:
        evict_least_frequently_used()
    if not os.path.exists(cache_path) and get_free_space_gb(CACHE_DIR) >= 5.0:
        cv2.imwrite(cache_path, image_data)
        increment_download_count(filename)

def clear_event_cache():
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except Exception:
                pass
    # Also clear meta
    meta_path = _get_meta_path()
    if os.path.exists(meta_path):
        try:
            os.remove(meta_path)
        except Exception:
            pass
