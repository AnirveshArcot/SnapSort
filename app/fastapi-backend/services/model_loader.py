# services/model_loader.py

from deepface import DeepFace
from ultralytics import YOLO

# Global models
model = None
arcface_model = None

def load_models():
    global model, arcface_model
    if model is None:
        model = YOLO("./yolov8n_face_trained.pt")
    if arcface_model is None:
        arcface_model = DeepFace.build_model("SFace")

def unload_models():
    global model, arcface_model
    try:
        import torch
        if model or arcface_model:
            torch.cuda.empty_cache()
        model = None
        arcface_model = None
    except ImportError:
        pass

def get_yolo_model():
    return model

def get_arcface_model():
    return arcface_model
