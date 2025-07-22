# db/mongo.py

from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)
db = client["snap-sort"]

users_collection = db["users"]
feature_vector_collection = db["image_feature_vectors"]
user_id_map = db["counters_collection"]
settings_coll = db["settings"]
