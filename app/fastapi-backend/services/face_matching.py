# services/face_matching.py
from services.image_processing import fetch_image_from_cdn
from tqdm import tqdm
import os
import json
from core.config import settings_coll, feature_vector_collection, user_id_map, CDN_STORAGE_PATH, CURRENT_EVENT_ID

def run_face_matching(faiss_index):
    matches = {}
    image_files = os.listdir(os.path.join(CDN_STORAGE_PATH, CURRENT_EVENT_ID))
    compressed_files = [f for f in image_files if f.endswith("_compressed.jpeg")]
    all_records = list(feature_vector_collection.find({"event_id": CURRENT_EVENT_ID}))
    int_id_to_obj = {doc["int_id"]: doc["_id"] for doc in user_id_map.find({})}

    for file_name in tqdm(compressed_files, desc="Matching Faces"):
        try:
            image = fetch_image_from_cdn(f"{CURRENT_EVENT_ID}/{file_name}")
            # call your vector matching logic here
            pass
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    with open(os.path.join(CDN_STORAGE_PATH, CURRENT_EVENT_ID, "matches.json"), "w") as f:
        json.dump({"matches": matches}, f, indent=4)

    settings_coll.update_one({"_id": "current_event"}, {"$set": {"status": "free"}}, upsert=True)