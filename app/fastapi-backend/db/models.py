# db/models.py
from pydantic import BaseModel
from typing import List, Optional

class User(BaseModel):
    id: str
    name: str
    email: str
    password: Optional[str] = None
    image: str
    joined_event: str
    role: str

class UserOut(BaseModel):
    id: str
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