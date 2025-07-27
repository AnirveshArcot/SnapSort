import base64
import re
from fastapi import BackgroundTasks, Cookie, FastAPI, Depends, HTTPException, Query, status, File, Form, Request, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import os
from datetime import datetime, timedelta
import jwt
import bcrypt
from pymongo import MongoClient, ReturnDocument
from bson import ObjectId
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import YOLO
import faiss
from deepface.DeepFace import represent
import numpy as np
import cv2
import json
import uvicorn
from secrets import token_urlsafe
from fastapi import APIRouter

from dotenv import load_dotenv

load_dotenv()



#Start-Up settings
faiss_index = None
CURRENT_EVENT_ID=None

async def lifespan(app: FastAPI):
    global faiss_index
    global CURRENT_EVENT_ID
    doc = settings_coll.find_one({"_id": "current_event"})
    if doc:
        CURRENT_EVENT_ID = str(doc["event_id"])
    else:
        # first‐time initialization
        settings_coll.insert_one({
            "_id": "current_event",
            "event_id": CURRENT_EVENT_ID
        })
    faiss_index = load_faiss_index(CURRENT_EVENT_ID, dimension)
    settings_coll.update_one(
        {"_id": "current_event"},
        {"$set": {"status": "free"}},
        upsert=True
    )
    yield


app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["snap-sort"]
users_collection = db["users"]
feature_vector_collection = db["image_feature_vectors"]
user_id_map = db["counters_collection"]
settings_coll = db["settings"]


#CDN Info
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




#Admin settings
ADMIN_MAIL=os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD=os.getenv("ADMIN_PASSWORD")






# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 

dimension = 4096
model = YOLO("./model.pt")


CDN_STORAGE_PATH = "./cdn_storage/"
CACHE_DIR="./image_cache/"
os.makedirs(CDN_STORAGE_PATH, exist_ok=True)
FAISS_INDEX_DIR = "./faiss_indices"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)



SIMILARITY_THRESHOLD=0.5

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

executor = ThreadPoolExecutor()

async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)

# Models

class ImageData(BaseModel):
    base64_images: List[str]


class User(BaseModel):
    id : str
    name: str
    email: str
    password: Optional[str] = None
    image: str
    joined_event: str
    role : str

class UserOut(BaseModel):
    id : str
    name: str
    email: str
    image: str
    joined_event: str
    role: str


class RegisterUser(BaseModel):
    name: str
    email: str
    password: str
    image: str

class Base64Image(BaseModel):
    filename: str
    base64: str

class UploadImagesRequest(BaseModel):
    images: List[Base64Image]

class UploadImagesResponse(BaseModel):
    uploaded: List[str]

class CreateUserRequest(BaseModel):
    name: str
    role: str

class DeleteUserRequest(BaseModel):
    email: str

    
def load_faiss_index(event_id: str, dimension: int):
    if event_id is None:
        raise ValueError("`event_id` must be provided")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    index_path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")

    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)
        print(f"Loaded FAISS index for event {event_id}")
    else:
        base = faiss.IndexFlatIP(dimension)
        faiss_index = faiss.IndexIDMap(base)
        print(f"Created new FAISS index for event {event_id}")

    return faiss_index

def save_faiss_index(event_id: str, dimension: int = None):
    if event_id is None:
        raise ValueError("`event_id` must be provided")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    if faiss_index is None:
        print(f"No FAISS index loaded for event {event_id}; nothing to save.")
        return

    index_path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    faiss.write_index(faiss_index, index_path)
    print(f"Saved FAISS index for event {event_id} to '{index_path}'")
    



def decode_base64_image(base64_string):
    base64_data = re.sub(r"^data:image/\w+;base64,", "", base64_string)
    image_data = base64.b64decode(base64_data)
    
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    return image

def fetch_image_from_cdn(image_name: str):
    _, ext = os.path.splitext(image_name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}")

    image_path = os.path.join(CDN_STORAGE_PATH, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {image_name}")

    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {image_name}")

    return image


def upload_to_cdn(file_name, json_data):
    file_path = os.path.join(CDN_STORAGE_PATH, file_name)

    try:
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save JSON file: {str(e)}")

    return {"url": f"local://{file_path}"}  # Simulated URL

def list_event_files(event_id: str):
    event_folder = os.path.join(CDN_STORAGE_PATH, event_id)
    if not os.path.exists(event_folder) or not os.path.isdir(event_folder):
        raise HTTPException(status_code=400, detail=f"Event folder not found: {event_id}")

    files = [
        fname for fname in os.listdir(event_folder)
        if os.path.isfile(os.path.join(event_folder, fname))
           and fname.lower().endswith(ALLOWED_EXTENSIONS)
    ]

    return {"event_id": event_id, "files": files}


def localize_faces_func(image):
    results = model.predict(source=image, conf=0.25,verbose=False)
    face_boxes = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        face_boxes.append((x1, y1, x2, y2))
    return face_boxes

def extract_features_func(face_image):
    result = represent(face_image, model_name="VGG-Face", enforce_detection=False, align=True)
    return result[0]["embedding"]

def normalize_vectors(vectors):
    """Normalize vectors to unit length for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / np.maximum(norms[:, np.newaxis], 1e-10)
    return normalized_vectors.astype('float32')

def allocate_int_id_for(uid):
    mapping = user_id_map.find_one({"_id": uid})
    if mapping:
        return mapping["int_id"]


    new_seq = user_id_map.find_one_and_update(
        {"_id": "user_id"},
        {"$inc": {"seq": 1}},
        return_document=ReturnDocument.AFTER,
        upsert=True
    )["seq"]


    user_id_map.insert_one({
        "_id": uid,
        "int_id": new_seq
    })

    return new_seq

    
def process_image(file, feature_records, int_id_map, event_id, similarity_threshold):
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

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=60))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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
        return User(
            id="NEO",
            name="ADMIN",
            email=ADMIN_MAIL,
            image="",
            joined_event=CURRENT_EVENT_ID,
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

    return User(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        image=user.get("image"),
        joined_event=CURRENT_EVENT_ID,
        role=user.get("role", "user")
    )

async def get_current_admin(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to perform this action"
        )
    return current_user




@app.post("/admin/match_faces")
async def match_faces(
    background_tasks: BackgroundTasks,
    admin_user: UserOut = Depends(get_current_admin)
):
    """Initiate face matching in the background and return immediately."""
    current_settings = settings_coll.find_one({"_id": "current_event"})
    current_status = current_settings.get("status") if current_settings else "free"
    if current_status == "processing":
        raise HTTPException(status_code=409, detail="Matching is already in progress.")
    settings_coll.update_one(
        {"_id": "current_event"},
        {"$set": {"status": "processing"}},
        upsert=True
    )
    background_tasks.add_task(run_face_matching)
    return {
        "message": "Face matching has started in the background.",
        "status": "processing"
    }


def run_face_matching():
    try:
        image_files = list_event_files(CURRENT_EVENT_ID)
        compressed_files = [
            file_name for file_name in image_files["files"]
            if os.path.splitext(file_name)[0].endswith("_compressed")
        ]
    
        matches = {}

        all_records = list(feature_vector_collection.find({"event_id": CURRENT_EVENT_ID}))
        if not all_records:
            print("No feature vectors found.")
            settings_coll.update_one(
            {"_id": "current_event"},
            {"$set": {"status": "error", "error_detail": str(e)}},
            upsert=True
        )
            return
        id_map_list = list(user_id_map.find({}, {"int_id": 1, "_id": 1}))
        int_id_to_obj = {record["int_id"]: record["_id"] for record in id_map_list if "int_id" in record}
        for file_name in tqdm(compressed_files, desc="Matching Faces"):
            base, ext = os.path.splitext(file_name)
            if not base.endswith("_compressed"):
                continue

            try:
                image_path = f"{CURRENT_EVENT_ID}/{file_name}"
                image = fetch_image_from_cdn(image_path)

                file = {"image": image, "file_key": file_name}
                result = process_image(file, all_records, int_id_to_obj, CURRENT_EVENT_ID, SIMILARITY_THRESHOLD)
                
                if result:
                    for person_id, file_matches in result.items():
                        if person_id not in matches:
                            matches[person_id] = []

                        for match in file_matches:
                            fk = match['file_key']
                            if fk not in matches[person_id]:
                                matches[person_id].append(fk)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        matches_json = {"matches": matches}
        matches_json_url = upload_to_cdn(f"{CURRENT_EVENT_ID}/matches.json", matches_json)
        settings_coll.update_one(
            {"_id": "current_event"},
            {"$set": {"status": "free"}},
            upsert=True
        )

    except Exception as e:
        settings_coll.update_one(
            {"_id": "current_event"},
            {"$set": {"status": "error", "error_detail": str(e)}},
            upsert=True
        )
        print(f"Error in background task: {e}")


@app.post("/auth/register")
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

    # Insert user and get MongoDB's _id
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

    save_faiss_index(CURRENT_EVENT_ID)

    return {
        "id": str(mongo_id),
        "name": user.name,
        "email": user.email,
        "image": user.image,
        "joined_event": CURRENT_EVENT_ID,
        "role": "user"
    }

    

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    if username == ADMIN_MAIL and password == ADMIN_PASSWORD:
        access_token = create_access_token(data={"sub": "admin", "role": "admin"})
        response = JSONResponse({"access_token": access_token, "token_type": "bearer"})
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            samesite="lax",
            secure=True,
            path="/"
        )
        return response
    user = users_collection.find_one({"email": username})
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": str(user["_id"]), "role": user["role"]})
    response = JSONResponse({"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="auth_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=True,
        path="/"
    )
    return response







@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(
    key="auth_token",
    path="/",              
    domain=None,
    )   
    return {"message": "Logged out"}

@app.get("/auth/me", response_model=UserOut)
async def get_me(current_user: UserOut = Depends(get_current_user)):
    return UserOut(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
        image=current_user.image,
        joined_event=CURRENT_EVENT_ID,
        role=current_user.role,
    )



@app.post("/event/upload-images", response_model=UploadImagesResponse, status_code=201)
async def event_upload_images(req: UploadImagesRequest, current_user: UserOut = Depends(get_current_user)):
    if current_user.role in ["admin", "photographer"]:
        event_folder = os.path.join(CDN_STORAGE_PATH, current_user.joined_event)
        os.makedirs(event_folder, exist_ok=True)
        uploaded = []
        for img in req.images:
            header, _, payload = img.base64.partition(",")
            try:
                image_data = base64.b64decode(payload or img.base64)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid base64 image for file {img.filename}: {e}"
                )
            base_name, ext = os.path.splitext(img.filename)
            original_filename = f"{base_name}_original{ext}"
            original_path = os.path.join(event_folder, original_filename)
            with open(original_path, "wb") as fout:
                fout.write(image_data)
            uploaded.append(original_filename)
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_np is None:
                    raise ValueError("Decoded image is None")
                preview = cv2.resize(img_np, (0, 0), fx=0.25, fy=0.25)
                preview_filename = f"{base_name}_preview.jpeg"
                preview_path = os.path.join(event_folder, preview_filename)
                quality = 60
                while True:
                    success, buffer = cv2.imencode(".jpeg", preview, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not success:
                        raise ValueError("Failed to encode preview image")
                    if len(buffer) <= 50 * 1024 or quality <= 30:
                        break
                    quality -= 5
                with open(preview_path, "wb") as fout:
                    fout.write(buffer)
                uploaded.append(preview_filename)
                compressed_filename = f"{base_name}_compressed.jpeg"
                compressed_path = os.path.join(event_folder, compressed_filename)
                quality = 80
                while True:
                    success, buffer = cv2.imencode(".jpeg", img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not success:
                        raise ValueError("Failed to encode compressed image")
                    if len(buffer) <= 512 * 1024 or quality <= 40:
                        break
                    quality -= 5
                with open(compressed_path, "wb") as fout:
                    fout.write(buffer)
                uploaded.append(compressed_filename)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate preview/compressed version for {img.filename}: {e}"
                )
        return UploadImagesResponse(uploaded=uploaded)
    elif current_user.role == "editor":
        event_folder = os.path.join(CDN_STORAGE_PATH, f"{current_user.joined_event}_edited")
        os.makedirs(event_folder, exist_ok=True)
        uploaded = []
        for img in req.images:
            header, _, payload = img.base64.partition(",")
            try:
                image_data = base64.b64decode(payload or img.base64)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid base64 image for file {img.filename}: {e}"
                )
            base_name, ext = os.path.splitext(img.filename)
            base_name = f"{base_name}_edited"
            filename = f"{base_name}{ext}"
            file_path = os.path.join(event_folder, filename)
            with open(file_path, "wb") as fout:
                fout.write(image_data)
            uploaded.append(filename)
        return UploadImagesResponse(uploaded=uploaded)
    else:
        raise HTTPException(status_code=403, detail="Not allowed to upload images")

@app.get("/event/get-images", response_model=List[dict])
async def get_event_images(current_user: UserOut = Depends(get_current_user)):
    event_folder = os.path.join(CDN_STORAGE_PATH, current_user.joined_event)
    images = []

    if not os.path.exists(event_folder):
        return images

    if current_user.role in ["admin", "editor"]:
        for f in os.listdir(event_folder):
            if f.lower().endswith("_preview.jpeg"):
                file_path = os.path.join(event_folder, f)
                try:
                    with open(file_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        images.append({
                            "name": f,
                            "base64": f"data:image/jpeg;base64,{encoded_string}"
                        })
                except Exception:
                    continue
        return images

    if current_user.role == "photographer":
        return []
    print(current_user.id)

    matches_path = os.path.join(event_folder, "matches.json")
    if not os.path.exists(matches_path):
        return images

    try:
        with open(matches_path, "r") as f:
            matches_data = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid matches.json format")

    matched_files = matches_data.get("matches", {}).get(current_user.id, [])

    for idx, filename in enumerate(matched_files):
        base, ext = os.path.splitext(filename)
        base = base.replace("_compressed", "")
        preview_filename = f"{base}_preview.jpeg"
        preview_path = os.path.join(event_folder, preview_filename)

        if not os.path.exists(preview_path):
            continue

        try:
            with open(preview_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                images.append({
                            "name": preview_filename,
                            "base64": f"data:image/jpeg;base64,{encoded_string}"
                        })
        except Exception:
            continue

    return images
    
    

@app.get("/event/download")
async def download_image(filename: str = Query(...), current_user: UserOut = Depends(get_current_user)):
    folder = os.path.join(CDN_STORAGE_PATH, current_user.joined_event)
    if "_preview" in filename:
        base_name = filename.replace("_preview", "").rsplit(".", 1)[0]
    else:
        base_name = os.path.splitext(filename)[0]
    if current_user.role == "editor":
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            original_filename = f"{base_name}_original{ext}"
            file_path = os.path.join(folder, original_filename)
            if os.path.exists(file_path):
                return FileResponse(path=file_path, filename=original_filename, media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail="File not found")
    else:
        compressed_filename = f"{base_name}_compressed.jpeg"
        file_path = os.path.join(folder, compressed_filename)
        if os.path.exists(file_path):
            return FileResponse(path=file_path, filename=compressed_filename, media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail="File not found")


class RoleUploadImagesRequest(BaseModel):
    images: List[Base64Image]

@app.post("/event/upload-images", response_model=UploadImagesResponse, status_code=201)
async def role_upload_event_images(req: RoleUploadImagesRequest, current_user: UserOut = Depends(get_current_user)):
    if current_user.role not in ["photographer", "editor"]:
        raise HTTPException(status_code=403, detail="Not allowed to upload images")
    if current_user.role == "editor":
        event_folder = os.path.join(CDN_STORAGE_PATH, f"{current_user.joined_event}_edited")
    else:
        event_folder = os.path.join(CDN_STORAGE_PATH, current_user.joined_event)
    os.makedirs(event_folder, exist_ok=True)
    uploaded = []
    for img in req.images:
        header, _, payload = img.base64.partition(",")
        try:
            image_data = base64.b64decode(payload or img.base64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid base64 image for file {img.filename}: {e}"
            )
        base_name, ext = os.path.splitext(img.filename)
        if current_user.role == "editor":
            base_name = f"{base_name}_edited"
        filename = f"{base_name}{ext}"
        file_path = os.path.join(event_folder, filename)
        with open(file_path, "wb") as fout:
            fout.write(image_data)
        uploaded.append(filename)
    return UploadImagesResponse(uploaded=uploaded)

@app.get("/event/role-download")
def role_download_image(filename: str = Query(...), current_user: UserOut = Depends(get_current_user)):
    if current_user.role != "editor":
        raise HTTPException(status_code=403, detail="Only editors can download images")
    folder = os.path.join(CDN_STORAGE_PATH, current_user.joined_event)
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"detail": "File not found"})
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.post("/admin/create-user")
def create_user(req: CreateUserRequest, admin: UserOut = Depends(get_current_admin)):
    if req.role not in ["photographer", "editor"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    email = f"{req.name}@arka.ai"
    password = token_urlsafe(8) 
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_data = {
        "name": req.name,
        "email": email,
        "password": hashed_password.decode('utf-8'),
        "role": req.role,
        "joined_event": CURRENT_EVENT_ID,
        "image": ""
    }
    if users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="User already exists")
    users_collection.insert_one(user_data)
    return {"email": email, "password": password, "role": req.role}

@app.post("/event/create-event")
async def create_event(admin: UserOut = Depends(get_current_admin)):
    global CURRENT_EVENT_ID

    new_id = str(ObjectId())

    settings_coll.update_one(
        {"_id": "current_event"},
        {"$set": {"event_id": new_id}},
        upsert=True
    )

    CURRENT_EVENT_ID = new_id

    users_collection.delete_many({})
    feature_vector_collection.delete_many({})
    user_id_map.delete_many({})

    event_folder_path = os.path.join(CDN_STORAGE_PATH, new_id)
    os.makedirs(event_folder_path, exist_ok=True)

    return {"event_id": new_id}


@app.get("/admin/list-users")
def list_users(admin: UserOut = Depends(get_current_admin)):
    users = list(users_collection.find({"role": {"$in": ["photographer", "editor"]}}))
    result = []
    for user in users:
        result.append({
            "name": user.get("name"),
            "email": user.get("email"),
            "role": user.get("role"),
            "password": user.get("password"),
        })
    return {"users": result}

@app.post("/admin/delete-user")
def delete_user(req: DeleteUserRequest, admin: UserOut = Depends(get_current_admin)):
    user = users_collection.find_one({"email": req.email, "role": {"$in": ["photographer", "editor"]}})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not allowed to delete")
    users_collection.delete_one({"_id": user["_id"]})
    return {"success": True, "email": req.email}




































@app.get("/")
async def root():
    return {"message": "Event Management API"}



def main():
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()