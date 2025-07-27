import os
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from db.models import CreateUserRequest, DeleteUserRequest
import core.config as config
from services.face_matching import run_face_matching
from core.security import create_access_token
from secrets import token_urlsafe
import bcrypt
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from services.image_processing import clear_event_cache
from .auth import get_me as get_current_user

router = APIRouter()

class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: str


async def get_current_admin(current_user: UserOut = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.post("/match_faces")
def match_faces(background_tasks: BackgroundTasks, admin_user: UserOut = Depends(get_current_admin)):
    current_settings = config.settings_coll.find_one({"_id": "current_event"})
    current_status = current_settings.get("status") if current_settings else "free"
    if current_status == "processing":
        raise HTTPException(status_code=409, detail="Matching is already in progress.")
    config.settings_coll.update_one(
        {"_id": "current_event"},
        {"$set": {"status": "processing"}},
        upsert=True
    )
    background_tasks.add_task(run_face_matching)
    return {"message": "Face matching has started in the background.", "status": "processing"}


@router.post("/create-user")
def create_user(req: CreateUserRequest, admin: UserOut = Depends(get_current_admin)):
    if req.role not in ["photographer", "editor"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    email = f"{req.name}@arka.ai"
    password = token_urlsafe(8)
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    if config.users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="User already exists")

    user_data = {
        "name": req.name,
        "email": email,
        "password": hashed_password.decode('utf-8'),
        "role": req.role,
        "joined_event": config.get_current_event_id(),
        "image": ""
    }
    
    config.users_collection.insert_one(user_data)
    return {"email": email, "password": password, "role": req.role}


@router.get("/list-users")
def list_users(admin: UserOut = Depends(get_current_admin)):
    users = list(config.users_collection.find({"role": {"$in": ["photographer", "editor"]}}))
    result = []
    for user in users:
        result.append({
            "name": user.get("name"),
            "email": user.get("email"),
            "role": user.get("role"),
        })
    return {"users": result}


@router.post("/delete-user")
def delete_user(req: DeleteUserRequest, admin: UserOut = Depends(get_current_admin)):
    user = config.users_collection.find_one({"email": req.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    config.users_collection.delete_one({"_id": user["_id"]})
    return {"success": True}


@router.post("/create-event")
def create_event(admin: UserOut = Depends(get_current_admin)):
    new_id = str(ObjectId())

    # Update event ID in DB and memory
    config.set_current_event_id(new_id)

    # Clear image match cache
    clear_event_cache()

    # Wipe old data
    config.users_collection.delete_many({})
    config.feature_vector_collection.delete_many({})
    config.user_id_map.delete_many({})

    # Create event folder in CDN
    event_folder_path = os.path.join(config.CDN_STORAGE_PATH, new_id)
    os.makedirs(event_folder_path, exist_ok=True)

    return {"event_id": new_id}
