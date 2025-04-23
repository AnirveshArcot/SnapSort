import base64
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


app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
client = MongoClient("mongodb+srv://anirvesh:anirvesh@cluster0.tuw5ikl.mongodb.net")
db = client["snap-sort"]
users_collection = db["users"]
events_collection = db["events"]
feature_vector_collection = db["image_feature_vectors"]
CDN_BASE_URL = "https://your-cdn.com/"
CDN_UPLOAD_URL = "https://your-cdn.com/upload"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# JWT settings
SECRET_KEY = "your-secret-key"  # Use a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

#FAISS and ML Settings
faiss_indices = {}
dimension = 4096
model = YOLO("model.pt")
FAISS_INDEX_DIR = "./faiss_indices"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Async wrapper for pymongo (since pymongo is synchronous)
executor = ThreadPoolExecutor()

async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)

# Models

class FaceMatchingRequest(BaseModel):
    similarity_threshold: float

class ImageData(BaseModel):
    base64_images: List[str]


class User(BaseModel):
    name: str
    email: str
    password: str
    image: Optional[str] = None
    joined_events: List[str]


class RegisterUser(BaseModel):
    name: str
    email: str
    password: str
    image: Optional[str] = None



class UserOut(BaseModel):
    id: str
    name: str
    email: str
    image: Optional[str] = None
    joined_events: List[str] = []

class Event(BaseModel):
    name: str
    description: str
    code: str
    created_at: datetime

class EventOut(BaseModel):
    id: str
    name: str
    description: str

class EventDetail(BaseModel):
    id: str
    name: str
    description: str
    code: str

class EventImage(BaseModel):
    event_id: str
    user_id: str
    url: str
    created_at: datetime

class EventImageOut(BaseModel):
    id: str
    url: str
    userName: str

class JoinEventRequest(BaseModel):
    code: str

# Helper functions
    
def load_faiss_index(event_id, dimension):
    index_path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Loaded FAISS index for event {event_id}")
    else:
        index = faiss.IndexFlatIP(dimension)  # Create a new index if not found
        print(f"Created new FAISS index for event {event_id}")
    return index

def save_faiss_index(event_id):
    index_path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    if event_id in faiss_indices:
        faiss.write_index(faiss_indices[event_id], index_path)
        print(f"Saved FAISS index for event {event_id}")

# def fetch_image_from_cdn(image_url):
#     response = requests.get(image_url)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail=f"Failed to fetch image: {image_url}")
    
#     image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
#     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#     if image is None:
#         raise HTTPException(status_code=400, detail=f"Failed to decode image: {image_url}")
    
#     return image


# def upload_to_cdn(file_name, json_data):
#     """Uploads JSON file to the CDN."""
#     files = {"file": (file_name, json.dumps(json_data), "application/json")}
#     response = requests.post(CDN_UPLOAD_URL, files=files)

#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Failed to upload matches.json to CDN")

#     return response.json().get("url")  # Assuming CDN returns the file URL
    

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

def list_event_files(event_id):
    """Lists all files in the given event's folder inside the local CDN storage."""
    event_folder = os.path.join(CDN_STORAGE_PATH, event_id)
    if not os.path.exists(event_folder) or not os.path.isdir(event_folder):
        raise HTTPException(status_code=400, detail=f"Event folder not found: {event_id}")

    files = os.listdir(event_folder)
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
            x, y, w, h = box
            face_image = image[y:y+h, x:x+w]
            feature_vector=extract_features_func(face_image)
            query_vector = np.array(feature_vector)
            normalized_query = normalize_vectors(query_vector.reshape(1, -1))
            k = 1 
            similarities, indices = faiss_indices[event_id].search(normalized_query, k)
            # Process results
            for sim, idx in zip(similarities[0], indices[0]):
                if sim >= 0:
                    person_id = str(all_records[idx]["_id"])
                    if person_id not in image_matches:
                        image_matches[person_id] = []
                    image_matches[person_id].append({
                        "file_key": file["file_key"],
                        "bounding_box": box,
                        "similarity": float(sim)
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
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = await run_in_threadpool(
        users_collection.find_one, {"_id": ObjectId(user_id)}
    )
    if not user:
        raise credentials_exception

    # build your pydantic output model
    return UserOut(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        image=user.get("image"),
        joined_events=user.get("joined_events", []),
    )

# Routes


@app.on_event("startup")
async def load_all_faiss_indices():
    global faiss_indices
    event_ids = events_collection.distinct("_id")
    for event_id in event_ids:
        str_event_id = str(event_id)
        faiss_indices[str_event_id] = load_faiss_index(str_event_id, dimension)




@app.post("/process_user_images/{event_id}")
async def process_user_images(event_id: str):
    stored_data = []
    try:
        event = events_collection.find_one({"_id": ObjectId(event_id)}, {"participants": 1})
        if not event or "participants" not in event:
            raise HTTPException(status_code=400, detail="Event not found or has no participants")
        
        user_ids = [ObjectId(uid) for uid in event["participants"]]  # Convert string IDs to ObjectId
        user_records = list(users_collection.find({"_id": {"$in": user_ids}}, {"_id": 1, "image": 1}))
        if not user_records:
            raise HTTPException(status_code=400, detail="No user images found")
        
        all_records = list(feature_vector_collection.find({"_id": {"$in": user_ids}}, {"_id": 1, "feature_vector": 1,}))

        existing_ids = {record["_id"] for record in all_records}
        
        new_vectors = []
        new_records = []
        for user in user_records:
            id = user["_id"]
            if id in existing_ids:
                continue
            base64_image = user["image"]
            image = decode_base64_image(base64_image)
            if image is None:
                raise ValueError(f"Invalid image data for user ID {id}")
            box = localize_faces_func(image)
            x, y, w, h = box[0]
            face_image = image[y:y+h, x:x+w]
            feature_vector = extract_features_func(face_image)

            record = {"feature_vector": np.array(feature_vector).tolist(), "_id": id , "event_id":event_id}
            feature_vector_collection.update_one(
                {"_id": id},
                {"$set": record},
                upsert=True
            )
            faiss_record= {"_id": str(id) , "event_id":event_id}
            new_vectors.append(feature_vector)
            stored_data.append(faiss_record)

        
        if new_vectors:
            if event_id not in faiss_indices:
                faiss_indices[event_id] = load_faiss_index(event_id, dimension)  # Load existing or create new

            index = faiss_indices[event_id]
            new_vectors_array = np.array(new_vectors)
            normalized_vectors = normalize_vectors(new_vectors_array)
            index.add(normalized_vectors)
            save_faiss_index(event_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing user images: {str(e)}")
    
    return {"message": "User images processed and data stored successfully", "stored_data": stored_data}

@app.post("/match_faces/{event_id}")
async def match_faces(event_id: str,request: FaceMatchingRequest):
    """Match faces from stored images for a specific event from CDN."""
    try:
        print("getting into match")
        # Construct CDN directory URL
        event_directory_url = f"{CDN_BASE_URL}{event_id}/"

        # Fetch image file list from CDN (Assuming a JSON API returns file names)
        # response = requests.get(event_directory_url)
        # if response.status_code != 200:
        #     raise HTTPException(status_code=400, detail="Event images directory not found on CDN")
        
        # image_files = response.json().get("images", [])  # Assuming CDN API returns {"images": ["image1.jpg", "image2.png"]}
        image_files = list_event_files(event_id)
        matches = {}

        print("Image files:", image_files["files"])  # Debugging: Ensure files are listed

        for file_name in image_files["files"]:
            try:
                image_path = f"{event_id}/{file_name}"
                image = fetch_image_from_cdn(image_path)

                file = {"image": image, "file_key": file_name}
                result = process_image(file, event_id, request.similarity_threshold)
                
                if result:
                    for person_id, file_matches in result.items():
                        if person_id not in matches:
                            matches[person_id] = []
                        matches[person_id].append([match['file_key'] for match in file_matches])
            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                

        matches_json = {"matches": matches}
        matches_json_url = upload_to_cdn(f"{event_id}/matches.json", matches_json)

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
        "joined_events":[],
    }
    result = users_collection.insert_one(user_data)
    return {
        "id": user_id,
        "name": user.name,
        "email": user.email,
    }

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await run_in_threadpool(users_collection.find_one, {"email": form_data.username})
    if not user or not bcrypt.checkpw(form_data.password.encode('utf-8'), user["password"].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": str(user["_id"])})
    response = JSONResponse({"access_token": access_token, "token_type": "bearer"})
    response.set_cookie("auth_token", access_token, httponly=True, secure=False, samesite="lax", path="/")
    return response

@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(
        key="auth_token",
        httponly=True,
        secure=True,
        samesite="strict",
        path="/"
    )
    return {"message": "Logged out", "ok": True}

@app.get("/auth/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserOut(
        id=current_user.id,
        name=current_user.name,
        email=current_user.email,
        image=current_user.image,
        joined_events=current_user.joined_events,
    )

@app.post("/events/join")
async def join_event(request: JoinEventRequest, current_user: UserOut = Depends(get_current_user)):
    
    event = events_collection.find_one({"code": request.code})
    if not event:
        raise HTTPException(status_code=404, detail="Invalid event code")
    
    event_id = str(event["_id"]) 
    user_id = str(current_user.id) 

    user = users_collection.find_one({"_id": ObjectId(current_user.id)})
    if user and "joined_events" in user and event_id in user["joined_events"]:
        raise HTTPException(status_code=400, detail="Already joined event")


    updated_user = users_collection.find_one_and_update(
        {"_id": ObjectId(current_user.id)},  
        {"$addToSet": {"joined_events": event_id}},  
        return_document=ReturnDocument.AFTER  
    )

    if not updated_user:
        raise HTTPException(status_code=500, detail="Failed to join event")

    events_collection.update_one(
        {"_id": ObjectId(event_id)},  
        {"$addToSet": {"participants": user_id}}
    )
    updated_user["_id"] = str(updated_user["_id"])  

    return {"message": "Successfully joined event", "user": updated_user}


@app.get("/events/list", response_model=List[EventOut])
async def get_joined_events(current_user: UserOut = Depends(get_current_user)):
    user = users_collection.find_one({"_id": ObjectId(current_user.id)})
    if not user or "joined_events" not in user or not user["joined_events"]:
        return []
    joined_event_ids = [ObjectId(eid) for eid in user["joined_events"]]
    events = list(events_collection.find({"_id": {"$in": joined_event_ids}}))
    joined_events = []
    for event in events:
        event_id_str = str(event["_id"])
        first_image = event.get("images", [None])[0]

        joined_events.append(EventOut(
            id=event_id_str,
            name=event["name"],
            description=event["description"],
            imageUrl=first_image["url"] if first_image else None,
        ))

    return joined_events

@app.get("/events/{event_id}")
async def get_event(event_id: str, current_user: UserOut = Depends(get_current_user)):
    """
    Fetch event details by event_id.
    """
    user = users_collection.find_one({"_id": ObjectId(current_user.id)})
    if not ObjectId.is_valid(event_id):
        raise HTTPException(status_code=400, detail="Invalid event ID format")

    if event_id not in user["joined_events"]:
        raise HTTPException(status_code=403, detail="Access denied. You have not joined this event.")
    
    event = events_collection.find_one({"_id": ObjectId(event_id)})
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    first_image = event.get("images", [None])[0]

    return EventOut(
        id=str(event["_id"]),
        name=event["name"],
        description=event["description"],
    )


































@app.get("/")
async def root():
    return {"message": "Event Management API"}



