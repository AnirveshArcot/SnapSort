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
import urllib
import onnxruntime


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


def get_feature_extractor():
    if not hasattr(get_feature_extractor, "_session"):
        model_dir = "./models"
        model_path = os.path.join(model_dir, "mobilefacenet.onnx")
        model_url = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcface_resnet100.onnx"

        os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(model_path):
            print("[Model] Downloading MobileFaceNet ONNX model...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("[Model] Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        get_feature_extractor._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    return get_feature_extractor._session



# ---------- Core Logic ----------

def extract_features_func(face_image: np.ndarray):
    try:
        model = get_feature_extractor()
        if face_image.shape[:2] != (112, 112):
            face_image = cv2.resize(face_image, (112, 112))
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_rgb = face_rgb.astype(np.float32)
        face_rgb = (face_rgb - 127.5) / 128.0  # normalize to [-1, 1]
        face_rgb = np.transpose(face_rgb, (2, 0, 1))[np.newaxis, ...]

        embedding = model.run(None, {"data": face_rgb})[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    except Exception as e:
        print(f"[extract_features_func] Error extracting features: {e}")
        return None


def localize_faces_func(image: np.ndarray):
    try:
        model = get_yolo_model()
        results = model.predict(source=image, conf=0.50, verbose=False)
        face_boxes = [
            tuple(map(int, box)) for box in results[0].boxes.xyxy
        ]
        return face_boxes
    except Exception as e:
        print(f"Error localizing faces: {e}")
        return []


def process_image(file, int_id_map, faiss_index, similarity_threshold):
    image = None
    vecs = []
    try:
        print("[process_image] Starting image processing...")

        image = file["image"]
        file_key = file["file_key"]
        print(f"[process_image] Processing file_key: {file_key}")

        bounding_boxes = localize_faces_func(image)
        print(f"[process_image] Found {len(bounding_boxes)} bounding boxes: {bounding_boxes}")

        if not bounding_boxes:
            print("[process_image] No faces detected. Returning empty result.")
            return {}

        valid_boxes = []

        for idx, (x1, y1, x2, y2) in enumerate(bounding_boxes):
            face_img = image[y1:y2, x1:x2]
            print(f"[process_image] Extracting features from face {idx+1} at box ({x1}, {y1}, {x2}, {y2})")
            print(face_img)
            feat = extract_features_func(face_img)
            if feat is not None:
                vec = np.array(feat, dtype='float32')
                vecs.append(vec)
                valid_boxes.append((x1, y1, x2, y2))
                print(f"[process_image] Feature vector extracted for face {idx+1}")
            else:
                print(f"[process_image] Feature extraction failed for face {idx+1}")

        if not vecs:
            print("[process_image] No valid feature vectors extracted. Returning empty result.")
            return {}

        print(f"[process_image] Stacking {len(vecs)} feature vectors into batch")
        batch = np.stack(vecs, axis=0)
        print("[process_image] Normalizing batch vectors using faiss.normalize_L2")
        faiss.normalize_L2(batch)

        print("[process_image] Searching FAISS index")
        similarities, indices = faiss_index.search(batch, 1)
        print(f"[process_image] Similarities: {similarities}")
        print(f"[process_image] Indices: {indices}")

        matches = {}
        for i, box in enumerate(valid_boxes):
            best_score = float(similarities[i, 0])
            best_int_id = int(indices[i, 0])
            print(f"[process_image] Face {i+1}: best_score = {best_score}, best_int_id = {best_int_id}")
            if best_score >= similarity_threshold:
                obj_id = int_id_map.get(best_int_id)
                if obj_id:
                    pid = str(obj_id)
                    print(f"[process_image] Match found: {pid} with score {best_score}")
                    matches.setdefault(pid, []).append({
                        "file_key": file_key,
                        "bounding_box": box,
                        "similarity": best_score
                    })
                else:
                    print(f"[process_image] No obj_id found for int_id {best_int_id}")
            else:
                print(f"[process_image] Similarity below threshold ({similarity_threshold})")

        print(f"[process_image] Total matches found: {len(matches)}")
        return matches

    except Exception as e:
        print(f"[process_image] Error: {e}")
        return None
    finally:
        print("[process_image] Cleaning up memory...")
        del image
        del file
        del vecs
        gc.collect()
        print("[process_image] Cleanup complete.")



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
        if hasattr(get_face_app, "_app"):
            del get_face_app._app
        gc.collect()
