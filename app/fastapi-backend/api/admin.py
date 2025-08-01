import os
import asyncio
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from secrets import token_urlsafe
import bcrypt

from db.models import CreateUserRequest, DeleteUserRequest
import core.config as config
from services.face_matching import run_face_matching
from core.security import create_access_token
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
async def match_faces(background_tasks: BackgroundTasks, admin_user: UserOut = Depends(get_current_admin)):
    current_settings = await config.settings_coll.find_one({"_id": "current_event"})
    current_status = current_settings.get("status") if current_settings else "free"
    if current_status == "processing":
        raise HTTPException(status_code=409, detail="Matching is already in progress.")
    
    await config.settings_coll.update_one(
        {"_id": "current_event"},
        {"$set": {"status": "processing"}},
        upsert=True
    )
    background_tasks.add_task(run_face_matching)
    return {"message": "Face matching has started in the background.", "status": "processing"}


@router.post("/create-user")
async def create_user(req: CreateUserRequest, admin: UserOut = Depends(get_current_admin)):
    if req.role not in ["photographer", "editor"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    email = f"{req.name}@arka.ai"
    password = token_urlsafe(8)
    hashed_password = await asyncio.to_thread(
        lambda: bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()
    )

    existing = await config.users_collection.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user_data = {
        "name": req.name,
        "email": email,
        "password": hashed_password,
        "role": req.role,
        "joined_event": config.get_current_event_id(),
        "image": ""
    }
    await config.users_collection.insert_one(user_data)
    return {"email": email, "password": password, "role": req.role}


@router.get("/list-users")
async def list_users(admin: UserOut = Depends(get_current_admin)):
    users_cursor = config.users_collection.find({"role": {"$in": ["photographer", "editor"]}})
    users = await users_cursor.to_list(length=1000)
    result = [
        {"name": user.get("name"), "email": user.get("email"), "role": user.get("role")}
        for user in users
    ]
    return {"users": result}


@router.post("/delete-user")
async def delete_user(req: DeleteUserRequest, admin: UserOut = Depends(get_current_admin)):
    user = await config.users_collection.find_one({"email": req.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await config.users_collection.delete_one({"_id": user["_id"]})
    return {"success": True}


@router.post("/create-event")
async def create_event(admin: UserOut = Depends(get_current_admin)):
    new_id = str(ObjectId())

    config.set_current_event_id(new_id)
    clear_event_cache()

    # Clear old collections
    await config.users_collection.delete_many({})
    await config.feature_vector_collection.delete_many({})
    await config.user_id_map.delete_many({})

    # Create event folder
    event_folder_path = os.path.join(config.IMAGE_STORAGE_PATH, new_id)
    os.makedirs(event_folder_path, exist_ok=True)

    return {"event_id": new_id}
