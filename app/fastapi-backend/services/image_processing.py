import base64
import re
import cv2
import numpy as np
import os
from fastapi import HTTPException
from core.config import ALLOWED_EXTENSIONS, CDN_STORAGE_PATH

def decode_base64_image(base64_string):
    base64_data = re.sub(r"^data:image/\w+;base64,", "", base64_string)
    image_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def fetch_image_from_cdn(image_name: str):
    path = os.path.join(CDN_STORAGE_PATH, image_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return cv2.imread(path)
