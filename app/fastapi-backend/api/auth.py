import asyncio
import base64
import gc
import os
import time
import cv2
import bcrypt, jwt, numpy as np
from bson import ObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Cookie
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
import psutil
from db.models import RegisterUser, UserOut
from core.config import (
    ALGORITHM, SECRET_KEY, users_collection, ADMIN_MAIL, ADMIN_PASSWORD,
    feature_vector_collection, get_current_event_id, get_faiss_index, set_faiss_index, dimension, user_id_map,settings_coll
)
from core.security import create_access_token
from services.image_processing import decode_base64_image
from services.face_matching import extract_features_func, localize_faces_func
from services.faiss_index import load_faiss_index, save_faiss_index

router = APIRouter()

def print_memory_usage(stage: str):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"[MEMORY] {stage}: {mem:.2f} MB")

def print_timing(stage: str, start_time: float):
    elapsed = time.perf_counter() - start_time
    print(f"[TIME] {stage}: {elapsed:.3f} seconds")
    return time.perf_counter()  # reset timer

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / np.maximum(norms[:, np.newaxis], 1e-10)
    return normalized_vectors.astype('float32')

async def allocate_int_id_for(uid):
    mapping = await user_id_map.find_one({"_id": uid})
    if mapping:
        return mapping["int_id"]
    new_seq_doc = await user_id_map.find_one_and_update(
        {"_id": "user_id"},
        {"$inc": {"seq": 1}},
        return_document=True,
        upsert=True
    )
    new_seq = new_seq_doc["seq"]
    await user_id_map.insert_one({"_id": uid, "int_id": new_seq})
    return new_seq

async def get_current_user(auth_token: str | None = Cookie(None)) -> UserOut:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if auth_token is None:
        raise credentials_exception

    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        role = payload.get("role", "user")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    if role == "admin" and user_id == "admin":
        return UserOut(
            id="NEO",
            name="ADMIN",
            email=ADMIN_MAIL,
            image="",
            joined_event="noeventbecauseadmin",
            role="admin"
        )

    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise credentials_exception

    return UserOut(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        image=user.get("image"),
        joined_event=await get_current_event_id(),
        role=user.get("role", "user")
    )

@router.post("/register", response_model=UserOut)
async def register_user(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    image: UploadFile = File(...)
):
    current_settings = await settings_coll.find_one({"_id": "current_event"})
    current_status = current_settings.get("status") if current_settings else "free"
    if current_status == "processing":
        raise HTTPException(status_code=409, detail="Face matching in progress. Please try again later.")

    existing_user = await users_collection.find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")

    try:
        image_bytes = await image.read()
        img_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    face_boxes = await asyncio.to_thread(localize_faces_func, img)
    if not face_boxes:
        raise HTTPException(status_code=400, detail="No face detected in image.")
    if len(face_boxes) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Upload an image with only one face.")

    # Compress image and encode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    success, encoded_img = cv2.imencode('.jpg', img, encode_param)
    if not success:
        raise HTTPException(status_code=500, detail="Image compression failed")
    compressed_base64 = (
    "data:image/jpeg;base64," + base64.b64encode(encoded_img).decode()
    )

    hashed_password = await asyncio.to_thread(
        lambda: bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    )

    current_event = await get_current_event_id()

    user_data = {
        "name": name,
        "email": email,
        "image": compressed_base64,
        "joined_event": current_event,
        "role": "user",
        "password": hashed_password,
    }

    result = await users_collection.insert_one(user_data)
    mongo_id = result.inserted_id

    async def process_vector():
        try:
            x, y, w, h = face_boxes[0]
            face_img = img[y:y + h, x:x + w]
            vec = await asyncio.to_thread(extract_features_func, face_img)
            if vec is None:
                raise ValueError("Feature vector extraction failed")

            int_id = await allocate_int_id_for(str(mongo_id))

            feature_record = {
                "_id": mongo_id,
                "feature_vector": np.array(vec).tolist(),
                "event_id": current_event
            }
            await feature_vector_collection.update_one(
                {"_id": mongo_id}, {"$set": feature_record}, upsert=True
            )

            faiss_index = await get_faiss_index()
            if faiss_index is None:
                print("[WARN] FAISS index not loaded for current event")
                return

            normed = normalize_vectors(np.array([vec]).astype("float32"))
            await asyncio.to_thread(
                lambda: faiss_index.add_with_ids(normed, np.array([int_id], dtype="int64"))
            )

            # ✅ This ensures memory + disk update in sync
            await set_faiss_index(faiss_index)

            del face_img, vec, normed
            gc.collect()

        except Exception as e:
            print(f"[ERROR] Background feature extraction failed: {e}")

    asyncio.create_task(process_vector())

    return UserOut(id=str(mongo_id), **user_data)


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

    if username == ADMIN_MAIL and password == ADMIN_PASSWORD:
        token = create_access_token({"sub": "admin", "role": "admin"})
        resp = JSONResponse({"access_token": token, "token_type": "bearer"})
        resp.set_cookie("auth_token", token, httponly=True, samesite="lax", secure=True, path="/")
        return resp

    user = await users_collection.find_one({"email": username})
    if not user or not await asyncio.to_thread(bcrypt.checkpw, password.encode(), user["password"].encode()):
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    token = create_access_token({"sub": str(user["_id"]), "role": user["role"]})
    resp = JSONResponse({"access_token": token, "token_type": "bearer"})
    resp.set_cookie("auth_token", token, httponly=True, samesite="lax", secure=True, path="/")
    return resp

@router.post("/logout")
async def logout():
    resp = JSONResponse({"message": "Logged out"})
    resp.delete_cookie("auth_token", path="/")
    return resp

@router.get("/me", response_model=UserOut)
async def get_me(current_user: UserOut = Depends(get_current_user)):
    return current_user
