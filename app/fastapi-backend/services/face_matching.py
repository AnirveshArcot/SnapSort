# services/face_matching.py
from fastapi import HTTPException
from services.image_processing import fetch_image_from_cdn
from tqdm import tqdm
import os
import json
import numpy as np
import faiss
from core.config import (
    settings_coll,
    feature_vector_collection,
    user_id_map,
    IMAGE_STORAGE_PATH,
    ALLOWED_EXTENSIONS,
    SIMILARITY_THRESHOLD,
    get_current_event_id,
    get_faiss_index
)
from deepface import DeepFace
from ultralytics import YOLO

model = YOLO("./model.pt")
arcface_model = DeepFace.build_model("ArcFace")

def extract_features_func(face_image):
    result = DeepFace.represent(
        face_image,
        model_name="ArcFace",
        enforce_detection=False,
        align=True
    )
    return result[0]["embedding"]


def localize_faces_func(image):
    results = model.predict(source=image, conf=0.25, verbose=False)
    face_boxes = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        face_boxes.append((x1, y1, x2, y2))
    return face_boxes

def process_image(file, feature_records, int_id_map, faiss_index, similarity_threshold):
    try:
        image = file["image"]
        file_key = file["file_key"]
        bounding_boxes = localize_faces_func(image)
        if not bounding_boxes:
            return {}

        vecs = []
        for (x1, y1, x2, y2) in bounding_boxes:
            face_img = image[y1:y2, x1:x2]
            feat = extract_features_func(face_img)
            vecs.append(np.array(feat, dtype='float32'))

        batch = np.stack(vecs, axis=0)
        faiss.normalize_L2(batch)
        similarities, indices = faiss_index.search(batch, 1)

        matches = {}
        for i, box in enumerate(bounding_boxes):
            best_score = float(similarities[i, 0])
            best_int_id = int(indices[i, 0])

            if best_score >= similarity_threshold:
                try:
                    obj_id = int_id_map[best_int_id]
                except KeyError:
                    continue

                pid = str(obj_id)
                matches.setdefault(pid, []).append({
                    "file_key": file_key,
                    "bounding_box": box,
                    "similarity": best_score
                })
        return matches
    except Exception as e:
        print(f"Error processing image : {e}")
        return None

def upload_to_cdn(file_name, json_data):
    file_path = os.path.join(IMAGE_STORAGE_PATH, file_name)

    try:
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save JSON file: {str(e)}")

    return {"url": f"local://{file_path}"}  # Simulated URL

def list_event_files(event_id: str):
    event_folder = os.path.join(IMAGE_STORAGE_PATH, event_id)
    if not os.path.exists(event_folder) or not os.path.isdir(event_folder):
        raise HTTPException(status_code=400, detail=f"Event folder not found: {event_id}")

    files = [
        fname for fname in os.listdir(event_folder)
        if os.path.isfile(os.path.join(event_folder, fname))
        and fname.lower().endswith(ALLOWED_EXTENSIONS)
    ]
    return {"event_id": event_id, "files": files}

async def run_face_matching():
    try:
        print("[INFO] Starting face matching task...")

        current_event = await get_current_event_id()
        print(f"[DEBUG] Current event ID: {current_event}")
        if not current_event:
            raise ValueError("No current event ID set")

        faiss_index = await get_faiss_index()
        if faiss_index is None:
            raise ValueError("FAISS index not loaded for current event")
        print("[INFO] FAISS index loaded successfully.")

        image_files = list_event_files(current_event)
        print(f"[DEBUG] Total files found: {len(image_files['files'])}")

        compressed_files = [
            file_name for file_name in image_files["files"]
            if os.path.splitext(file_name)[0].endswith("_compressed")
        ]
        print(f"[INFO] Compressed files to process: {len(compressed_files)}")

        matches = {}

        all_records_cursor = feature_vector_collection.find({"event_id": current_event})
        all_records = await all_records_cursor.to_list(length=None)
        print(f"[DEBUG] Feature vectors loaded: {len(all_records)}")

        if not all_records:
            print("No feature vectors found.")
            settings_coll.update_one(
                {"_id": "current_event"},
                {"$set": {"status": "error", "error_detail": "No feature vectors found."}},
                upsert=True
            )
            return

        map_cursor = user_id_map.find({}, {"int_id": 1, "_id": 1})
        id_map_list = await map_cursor.to_list(length=None)
        print(f"[DEBUG] ID map records loaded: {len(id_map_list)}")

        int_id_to_obj = {
            record["int_id"]: record["_id"]
            for record in id_map_list if "int_id" in record
        }

        for file_name in tqdm(compressed_files, desc="Matching Faces"):
            base, ext = os.path.splitext(file_name)
            if not base.endswith("_compressed"):
                continue

            try:
                image_path = f"{current_event}/{file_name}"
                print(f"[INFO] Processing file: {image_path}")
                image = fetch_image_from_cdn(image_path)

                file = {"image": image, "file_key": file_name}
                result = process_image(file, all_records, int_id_to_obj, faiss_index, SIMILARITY_THRESHOLD)

                if result:
                    for person_id, file_matches in result.items():
                        if person_id not in matches:
                            matches[person_id] = []

                        for match in file_matches:
                            fk = match['file_key']
                            if fk not in matches[person_id]:
                                matches[person_id].append(fk)

                    print(f"[DEBUG] Matches found in {file_name}: {result}")
                else:
                    print(f"[DEBUG] No matches found for {file_name}")

            except Exception as e:
                print(f"[ERROR] Error processing {file_name}: {e}")

        matches_json = {"matches": matches}
        print(f"[INFO] Uploading matches.json to CDN: {current_event}/matches.json")
        upload_to_cdn(f"{current_event}/matches.json", matches_json)

        settings_coll.update_one(
            {"_id": "current_event"},
            {"$set": {"status": "free"}},
            upsert=True
        )
        print("[INFO] Face matching task completed successfully.")

    except Exception as e:
        print(f"[FATAL] Error in background task: {e}")
        settings_coll.update_one(
            {"_id": "current_event"},
            {"$set": {"status": "error", "error_detail": str(e)}},
            upsert=True
        )
