# api/admin.py
from fastapi import APIRouter, Depends, HTTPException
from db.models import CreateUserRequest, DeleteUserRequest
from core.config import users_collection, user_id_map, settings_coll, CURRENT_EVENT_ID
from core.security import create_access_token
from secrets import token_urlsafe
import bcrypt

router = APIRouter()

@router.post("/create-user")
def create_user(req: CreateUserRequest):
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

@router.get("/list-users")
def list_users():
    users = list(users_collection.find({"role": {"$in": ["photographer", "editor"]}}))
    return [{"name": u["name"], "email": u["email"], "role": u["role"]} for u in users]

@router.post("/delete-user")
def delete_user(req: DeleteUserRequest):
    user = users_collection.find_one({"email": req.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    users_collection.delete_one({"_id": user["_id"]})
    return {"success": True}

@router.post("/create-event")
def create_event():
    from bson import ObjectId
    new_id = str(ObjectId())
    settings_coll.update_one({"_id": "current_event"}, {"$set": {"event_id": new_id}}, upsert=True)
    return {"event_id": new_id}