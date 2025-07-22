# api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Cookie
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from db.models import RegisterUser, UserOut
from core.config import users_collection, CURRENT_EVENT_ID, ADMIN_MAIL, ADMIN_PASSWORD
from core.security import create_access_token
from services.image_processing import decode_base64_image
import bcrypt, jwt, numpy as np
import cv2
from bson import ObjectId
from services.faiss_index import save_faiss_index

router = APIRouter()

@router.post("/register", response_model=UserOut)
def register_user(user: RegisterUser):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": hashed_password.decode('utf-8'),
        "image": user.image,
        "joined_event": CURRENT_EVENT_ID,
        "role": "user"
    }
    result = users_collection.insert_one(user_data)
    return UserOut(id=str(result.inserted_id), **user_data)

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    if username == ADMIN_MAIL and password == ADMIN_PASSWORD:
        token = create_access_token({"sub": "admin", "role": "admin"})
        resp = JSONResponse({"access_token": token, "token_type": "bearer"})
        resp.set_cookie("auth_token", token, httponly=True)
        return resp
    user = users_collection.find_one({"email": username})
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    token = create_access_token({"sub": str(user["_id"]), "role": user["role"]})
    resp = JSONResponse({"access_token": token, "token_type": "bearer"})
    resp.set_cookie("auth_token", token, httponly=True)
    return resp

@router.post("/logout")
def logout():
    resp = JSONResponse({"message": "Logged out"})
    resp.delete_cookie("auth_token")
    return resp

@router.get("/me", response_model=UserOut)
def get_me(auth_token: str = Cookie(None)):
    if not auth_token:
        raise HTTPException(status_code=401, detail="No auth token")
    try:
        payload = jwt.decode(auth_token, ADMIN_PASSWORD, algorithms=["HS256"])
        user_id = payload.get("sub")
        if payload.get("role") == "admin":
            return UserOut(id="admin", name="ADMIN", email=ADMIN_MAIL, image="", joined_event=CURRENT_EVENT_ID, role="admin")
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        return UserOut(
            id=str(user["_id"]),
            name=user["name"],
            email=user["email"],
            image=user.get("image"),
            joined_event=user.get("joined_event"),
            role=user.get("role")
        )
    except:
        raise HTTPException(status_code=401, detail="Invalid token")