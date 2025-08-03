# services/face_matching.py
import asyncio
import gc
import traceback
import os
import json
import cv2
import numpy as np
import faiss
from tqdm import tqdm
from fastapi import HTTPException

from services.image_processing import fetch_image_from_cdn
from core.config import (
    settings_coll,
    feature_vector_collection,
    user_id_map,
    IMAGE_STORAGE_PATH,
    ALLOWED_EXTENSIONS,
    SIMILARITY_THRESHOLD,
    get_current_event_id,
    get_faiss_index,
)


# ---------- Models (Load-once) ----------

def get_yolo_model():
    from ultralytics import YOLO
    if not hasattr(get_yolo_model, "_model"):
        get_yolo_model._model = YOLO("./yolov8n_face_trained.pt")
    return get_yolo_model._model


def get_openface_model():
    from deepface.basemodels import OpenFace
    if not hasattr(get_openface_model, "_model"):
        get_openface_model._model = OpenFace.loadModel()
    return get_openface_model._model


# ---------- Feature Extractor ----------

def extract_features_func(face_image: np.ndarray):
    from deepface.commons import functions
    try:
        model = get_openface_model()
        target_size = (96, 96)
        preprocessed_img = functions.preprocess_face(
            img=face_image,
            target_size=target_size,
            enforce_detection=False,
            detector_backend="skip",
            align=True
        )
        embedding = model.predict(preprocessed_img)[0].tolist()
        return embedding
    except Exception:
        return None


# ---------- Face Localization ----------

def localize_faces_func(image: np.ndarray):
    try:
        model = get_yolo_model()
        results = model.predict(source=image, conf=0.50, verbose=False)
        face_boxes = [tuple(map(int, box)) for box in results[0].boxes.xyxy]
        return face_boxes
    except Exception:
        return []


# ---------- Face Processing ----------

def process_image(file, int_id_map, faiss_index, similarity_threshold):
    image = None
    vecs = []
    try:
        image = file["image"]
        file_key = file["file_key"]
        bounding_boxes = localize_faces_func(image)
        if not bounding_boxes:
            return {}

        valid_boxes = []
        for x1, y1, x2, y2 in bounding_boxes:
            face_img = image[y1:y2, x1:x2]
            feat = extract_features_func(face_img)
            if feat is not None:
                vecs.append(np.array(feat, dtype='float32'))
                valid_boxes.append((x1, y1, x2, y2))

        if not vecs:
            return {}

        batch = np.stack(vecs, axis=0)
        faiss.normalize_L2(batch)
        similarities, indices = faiss_index.search(batch, 1)

        matches = {}
        for i, box in enumerate(valid_boxes):
            score = float(similarities[i, 0])
            int_id = int(indices[i, 0])
            if score >= similarity_threshold:
                obj_id = int_id_map.get(int_id)
                if obj_id:
                    pid = str(obj_id)
                    matches.setdefault(pid, []).append({
                        "file_key": file_key,
                        "bounding_box": box,
                        "similarity": score
                    })
        return matches

    except Exception:
        return None
    finally:
        del image
        del file
        del vecs
        gc.collect()


# ---------- Utility ----------

def upload_to_cdn(file_name: str, json_data: dict):
    file_path = os.path.join(IMAGE_STORAGE_PATH, file_name)
    try:
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save JSON file: {str(e)}")
    return {"url": f"local://{file_path}"}


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


# ---------- Main Matching Logic ----------

async def run_face_matching():
    try:
        current_event = await get_current_event_id()
        if not current_event:
            raise ValueError("No current event ID set")

        faiss_index = await get_faiss_index()
        if faiss_index is None:
            raise ValueError("FAISS index not loaded for current event")

        image_files = list_event_files(current_event)
        compressed_files = [
            fname for fname in image_files["files"]
            if os.path.splitext(fname)[0].endswith("_compressed")
        ]

        all_records = await feature_vector_collection.find({"event_id": current_event}).to_list(length=None)
        if not all_records:
            await settings_coll.update_one(
                {"_id": "current_event"},
                {"$set": {"status": "error", "error_detail": "No feature vectors found."}},
                upsert=True
            )
            return

        id_map_list = await user_id_map.find({}, {"int_id": 1, "_id": 1}).to_list(length=None)
        int_id_to_obj = {
            record["int_id"]: record["_id"] for record in id_map_list if "int_id" in record
        }

        matches = {}

        for file_name in tqdm(compressed_files, desc="Matching Faces"):
            try:
                image_path = f"{current_event}/{file_name}"
                image = await asyncio.to_thread(fetch_image_from_cdn, image_path)
                if image is None:
                    continue

                file = {"image": image, "file_key": file_name}
                result = await asyncio.to_thread(
                    process_image, file, int_id_to_obj, faiss_index, SIMILARITY_THRESHOLD
                )

                if result:
                    for person_id, file_matches in result.items():
                        matches.setdefault(person_id, [])
                        for match in file_matches:
                            if match['file_key'] not in matches[person_id]:
                                matches[person_id].append(match['file_key'])

            except Exception:
                traceback.print_exc()
                continue
            finally:
                gc.collect()

        await asyncio.to_thread(upload_to_cdn, f"{current_event}/matches.json", {"matches": matches})

        await settings_coll.update_one(
            {"_id": "current_event"},
            {"$set": {"status": "free"}},
            upsert=True
        )

    except Exception as e:
        await settings_coll.update_one(
            {"_id": "current_event"},
            {"$set": {"status": "error", "error_detail": str(e)}},
            upsert=True
        )

    finally:
        if hasattr(get_yolo_model, "_model"):
            del get_yolo_model._model
        if hasattr(get_openface_model, "_model"):
            del get_openface_model._model
        gc.collect()
