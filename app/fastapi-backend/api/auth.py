# api/auth.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException, status, Cookie
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from db.models import RegisterUser, User, UserOut
from core.config import ALGORITHM, SECRET_KEY, users_collection, CURRENT_EVENT_ID, ADMIN_MAIL, ADMIN_PASSWORD
from core.security import create_access_token
from services.image_processing import decode_base64_image
import bcrypt, jwt, numpy as np
import cv2
from bson import ObjectId
from services.faiss_index import save_faiss_index
from services.face_matching import extract_features_func, localize_faces_func
from core.config import feature_vector_collection, faiss_index, dimension
from services.faiss_index import load_faiss_index, save_faiss_index

router = APIRouter()

def allocate_int_id_for(uid):
    from core.config import user_id_map
    mapping = user_id_map.find_one({"_id": uid})
    if mapping:
        return mapping["int_id"]
    new_seq = user_id_map.find_one_and_update(
        {"_id": "user_id"},
        {"$inc": {"seq": 1}},
        return_document=True,
        upsert=True
    )["seq"]
    user_id_map.insert_one({"_id": uid, "int_id": new_seq})
    return new_seq

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / np.maximum(norms[:, np.newaxis], 1e-10)
    return normalized_vectors.astype('float32')

executor = ThreadPoolExecutor()
async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)

async def get_current_user(
    auth_token: str | None = Cookie(None),
) -> UserOut:
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

    try:
        user = await run_in_threadpool(
            users_collection.find_one, {"_id": ObjectId(user_id)}
        )
    except Exception:
        raise credentials_exception

    if not user:
        raise credentials_exception

    return UserOut(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        image=user.get("image"),
        joined_event=CURRENT_EVENT_ID,
        role=user.get("role", "user")
    )

@router.post("/register", response_model=UserOut)
def register_user(user: RegisterUser):
    global faiss_index
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    try:
        img = decode_base64_image(user.image)
        if img is None:
            raise ValueError("Invalid image data")
        box = localize_faces_func(img)
        if not box:
            raise ValueError("No face detected in the image")
        x, y, w, h = box[0]
        face_img = img[y:y + h, x:x + w]
        vec = extract_features_func(face_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": hashed_password.decode('utf-8'),
        "image": user.image,
        "joined_event": CURRENT_EVENT_ID,
        "role": "user"
    }
    result = users_collection.insert_one(user_data)
    mongo_id = result.inserted_id
    int_id = allocate_int_id_for(str(mongo_id))
    feature_record = {
        "_id": mongo_id,
        "feature_vector": np.array(vec).tolist(),
        "event_id": CURRENT_EVENT_ID
    }
    feature_vector_collection.update_one(
        {"_id": mongo_id},
        {"$set": feature_record},
        upsert=True
    )
    if faiss_index is None:
        faiss_index = load_faiss_index(CURRENT_EVENT_ID, dimension)
    normed = normalize_vectors(np.array([vec]).astype("float32"))
    faiss_index.add_with_ids(normed, np.array([int_id], dtype="int64"))
    save_faiss_index(faiss_index, CURRENT_EVENT_ID)
    return UserOut(id=str(mongo_id), **user_data)

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    if username == ADMIN_MAIL and password == ADMIN_PASSWORD:
        token = create_access_token({"sub": "admin", "role": "admin"})
        resp = JSONResponse({"access_token": token, "token_type": "bearer"})
        resp.set_cookie(
            key="auth_token",
            value=token,
            httponly=True,
            samesite="lax",
            secure=True,
            path="/"
        )
        return resp
    user = users_collection.find_one({"email": username})
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    token = create_access_token({"sub": str(user["_id"]), "role": user["role"]})
    resp = JSONResponse({"access_token": token, "token_type": "bearer"})
    resp.set_cookie(
            key="auth_token",
            value=token,
            httponly=True,
            samesite="lax",
            secure=True,
            path="/"
        )
    return resp

@router.post("/logout")
def logout():
    resp = JSONResponse({"message": "Logged out"})
    resp.delete_cookie("auth_token", path="/")
    return resp


@router.get("/me", response_model=UserOut)
async def get_me(current_user: UserOut = Depends(get_current_user)):
    return UserOut(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
        image=current_user.image,
        joined_event=CURRENT_EVENT_ID,
        role=current_user.role,
    )
    

    
