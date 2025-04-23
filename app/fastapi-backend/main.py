import base64
from contextlib import asynccontextmanager
import io
import re
from PIL import Image
from fastapi import Cookie, FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
from datetime import datetime, timedelta
import jwt
import bcrypt
from passlib.context import CryptContext
from pymongo import MongoClient, ReturnDocument
from bson import ObjectId
import asyncio
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from ultralytics import YOLO
import faiss
from deepface.DeepFace import represent
import numpy as np
import requests
import cv2
import json




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
    yield


app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.186.125:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
client = MongoClient("mongodb+srv://anirvesh:anirvesh@cluster0.tuw5ikl.mongodb.net")
db = client["snap-sort"]
users_collection = db["users"]
feature_vector_collection = db["image_feature_vectors"]
user_id_map = db["counters_collection"]
settings_coll = db["settings"]


#CDN Info
CDN_BASE_URL = "https://your-cdn.com/"
CDN_UPLOAD_URL = "https://your-cdn.com/upload"
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




#Admin settings
ADMIN_MAIL="harsha@arka.ai"
ADMIN_PASSWORD="arkaai"






# JWT settings
SECRET_KEY = "your-secret-key"  # Use a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

#FAISS and ML Settings
dimension = 4096
model = YOLO("model.pt")
FAISS_INDEX_DIR = "./faiss_indices"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
SIMILARITY_THRESHOLD=0.25

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Async wrapper for pymongo (since pymongo is synchronous)
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

class UploadImagesRequest(BaseModel):
    images: List[str]

class UploadImagesResponse(BaseModel):
    uploaded: List[str]

# Helper functions
    
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
    

# Simulated CDN Storage Path
CDN_STORAGE_PATH = "./cdn_storage/"
os.makedirs(CDN_STORAGE_PATH, exist_ok=True)

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
    """Simulates uploading a JSON file to the local CDN folder."""
    file_path = os.path.join(CDN_STORAGE_PATH, file_name)

    try:
        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save JSON file: {str(e)}")

    return {"url": f"local://{file_path}"}  # Simulated URL

def list_event_files(event_id: str):
    """Lists only image files in the given event's folder inside the local CDN storage."""
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
    results = model.predict(source=image, conf=0.25)
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

def get_object_id_from_int(int_id) -> ObjectId:
    mapping = user_id_map.find_one(
        {"int_id": int_id},
        {"_id": 1}
    )
    if not mapping:
        raise KeyError(f"No ObjectId found for int_id={int_id}")
    return mapping["_id"]
    
def process_image(file, event_id, similarity_threshold):
    try:
        image = file["image"]
        print(file["file_key"])
        if image is None:
            print(f"Failed to read image: {file}")
            return None

        bounding_boxes = localize_faces_func(image)
        image_matches = {}

        # Get feature vectors only for the given event_id
        all_records = list(feature_vector_collection.find({"event_id": event_id}))

        # If no feature vectors are found for the event, return empty matches
        if not all_records:
            return image_matches

        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            face_image = image[y1:y2, x1:x2]
            feature_vector=extract_features_func(face_image)
            query_vector = np.array(feature_vector)
            normalized_query = normalize_vectors(query_vector.reshape(1, -1))
            k = 1 
            similarities, indices = faiss_index.search(normalized_query, k)
            best_score = float(similarities[0, 0])
            best_int_id = int(indices[0, 0])
            best_obj_id = get_object_id_from_int(best_int_id)
            # Process results
            if best_score >= similarity_threshold:
                    person_id = str(best_obj_id)
                    if person_id not in image_matches:
                        image_matches[person_id] = []
                    # print({
                    #     "file_key": file["file_key"],
                    #     "bounding_box": box,
                    #     "similarity": float(best_score)
                    # })
                    image_matches[person_id].append({
                        "file_key": file["file_key"],
                        "bounding_box": box,
                        "similarity": float(best_score)
                    })
        return image_matches
    except Exception as e:
        print(f"Error processing image : {e}")
        return None

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
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

    # Admin user logic
    if role == "admin" and user_id == "admin":
        return User(
            id="NEO",
            name="ADMIN",
            email=ADMIN_MAIL,
            image="",
            joined_event="ADMIN",
            role="admin"
        )

    # Regular user logic
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
        joined_event=user.get("joined_event"),
        role=user.get("role", "user")
    )

async def get_current_admin(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to perform this action"
        )
    return current_user


# Routes

@app.post("/admin/process_user_images")
async def process_user_images(
    admin_user: UserOut = Depends(get_current_admin)
):
    global faiss_index
    stored_data = []
    try:

        user_records = list(
            users_collection.find(
                {},
                {"_id": 1, "image": 1}
            )
        )
        if not user_records:
            raise HTTPException(status_code=400, detail="No users found in collection")

        user_ids = [user["_id"] for user in user_records]
        existing = list(
            feature_vector_collection.find(
                {"_id": {"$in": user_ids}},
                {"_id": 1}
            )
        )
        existing_ids = {rec["_id"] for rec in existing}

        new_vectors = []
        new_ids = []
        for user in user_records:
            uid = user["_id"]
            if uid in existing_ids:
                continue

            img = decode_base64_image(user["image"])
            if img is None:
                raise ValueError(f"Invalid image data for user ID {uid}")


            box = localize_faces_func(img)
            x, y, w, h = box[0]
            face_img = img[y:y+h, x:x+w]


            vec = extract_features_func(face_img)

            new_vectors.append(vec)
            int_id = allocate_int_id_for(uid)
            new_ids.append(int_id)

            record = {
                "_id": uid,
                "feature_vector": np.array(vec).tolist(),
                "event_id": CURRENT_EVENT_ID
            }

            feature_vector_collection.update_one(
                {"_id": uid},
                {"$set": record},
                upsert=True
            )
            stored_data.append({"_id": str(uid), "event_id": CURRENT_EVENT_ID})


        if new_vectors:
            if faiss_index is None:
                faiss_index = load_faiss_index(CURRENT_EVENT_ID, dimension)
            arr = np.vstack(new_vectors).astype('float32')
            ids = np.array(new_ids, dtype='int64')

            normed = normalize_vectors(arr)
            faiss_index.add_with_ids(normed, ids)
            save_faiss_index(CURRENT_EVENT_ID)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing user images: {e}")

    return {"message": "User images processed for all users successfully", "stored_data": stored_data}




@app.post("/admin/match_faces")
async def match_faces(
    admin_user: UserOut = Depends(get_current_admin)
):
    """Match faces from stored images for a specific event from CDN."""
    try:
        # Construct CDN directory URL
        event_directory_url = f"{CDN_BASE_URL}{CURRENT_EVENT_ID}/"

        # Fetch image file list from CDN (Assuming a JSON API returns file names)
        # response = requests.get(event_directory_url)
        # if response.status_code != 200:
        #     raise HTTPException(status_code=400, detail="Event images directory not found on CDN")
        
        # image_files = response.json().get("images", [])  # Assuming CDN API returns {"images": ["image1.jpg", "image2.png"]}
        image_files = list_event_files(CURRENT_EVENT_ID)
        matches = {}

        print("Image files:", image_files["files"])  # Debugging: Ensure files are listed

        for file_name in image_files["files"]:
            try:
                image_path = f"{CURRENT_EVENT_ID}/{file_name}"
                image = fetch_image_from_cdn(image_path)

                file = {"image": image, "file_key": file_name}
                result = process_image(file, CURRENT_EVENT_ID, SIMILARITY_THRESHOLD)
                
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

        return {
            "message": "Face matching completed successfully",
            "matches_file_url": matches_json_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/auth/register")
def register_user(user: RegisterUser):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    user_id = str(uuid4())
    user_data = {
        "id": user_id,
        "name": user.name,
        "email": user.email,
        "password": hashed_password.decode('utf-8'),
        "image": user.image,
        "joined_event":CURRENT_EVENT_ID,
        "role":"role",
    }
    result = users_collection.insert_one(user_data)
    return {
        "id": user_id,
        "name": user.name,
        "email": user.email,
        "image": user.image,
        "joined_event":CURRENT_EVENT_ID,
        "role":"user"
    }

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await run_in_threadpool(users_collection.find_one, {"email": form_data.username})
    if not user or not bcrypt.checkpw(form_data.password.encode('utf-8'), user["password"].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": str(user["_id"]), "role": "user"})
    response = JSONResponse({"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
    key="auth_token",
    value=access_token,
    httponly=True,
    samesite="lax",    
    secure=False,
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
        joined_event=current_user.joined_event,
        role=current_user.role,
    )





@app.get("/event/get-images", response_model=List[dict])
async def get_event_images(current_user: UserOut = Depends(get_current_user)):
    event_folder = os.path.join(CDN_STORAGE_PATH, current_user.joined_event)
    matches_path = os.path.join(event_folder, "matches.json")
    if not os.path.exists(event_folder):
        return []
    

    if not os.path.exists(matches_path):
        return []

    try:
        with open(matches_path, "r") as f:
            matches_data = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid matches.json format")

    image_dicts = []
    if current_user.id not in matches_data['matches']:
        return image_dicts
    for idx, filename in enumerate(matches_data['matches'][current_user.id]):
        _, ext = os.path.splitext(filename)
        if ext.lower() in ALLOWED_EXTENSIONS:
            image_path = os.path.join(event_folder, filename)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is None:
                    continue
                success, buffer = cv2.imencode(ext, img)
                if not success:
                    continue
                encoded_string = base64.b64encode(buffer).decode("utf-8")
                image_dicts.append({
                    "id": idx,
                    "image_base64": f"data:image/{ext[1:]};base64,{encoded_string}"
                })

    return image_dicts

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
    
    return {"event_id": new_id}


@app.post(
    "/event/upload-images",
    response_model=UploadImagesResponse,
    status_code=201
)
async def upload_event_images(
    req: UploadImagesRequest,
    admin: UserOut = Depends(get_current_admin)
):
    event_folder = os.path.join(CDN_STORAGE_PATH, CURRENT_EVENT_ID)
    os.makedirs(event_folder, exist_ok=True)

    uploaded = []
    for b64 in req.images:
        header, _, payload = b64.partition(",")
        try:
            data = base64.b64decode(payload or b64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid base64 image: {e}"
            )

        file_key = f"{uuid.uuid4().hex}.jpg"
        dest = os.path.join(event_folder, file_key)
        with open(dest, "wb") as fout:
            fout.write(data)

        uploaded.append(file_key)
    return UploadImagesResponse(uploaded=uploaded)


@app.post("/auth/admin/login")
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_MAIL or form_data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    access_token = create_access_token(data={"sub": "admin", "role": "admin"})
    response = JSONResponse({"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="auth_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/"
    )
    return response







































@app.get("/")
async def root():
    return {"message": "Event Management API"}



