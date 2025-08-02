# services/model_loader.py
import time
import gc
import threading
from typing import Optional
from deepface import DeepFace
from ultralytics import YOLO

# Global models with lazy loading
_yolo_model = None
_last_yolo_used = 0
_model_lock = threading.Lock()

# Configuration for model management
MODEL_CACHE_TTL = 300 
MEMORY_THRESHOLD = 0.8

def _get_memory_usage():
    try:
        import psutil
        return psutil.virtual_memory().percent / 100.0
    except ImportError:
        return 0.5  

def _should_unload_model(last_used: float) -> bool:
    if time.time() - last_used > MODEL_CACHE_TTL:
        return True
    if _get_memory_usage() > MEMORY_THRESHOLD:
        return True
    return False

def _unload_yolo_model():
    """Safely unload YOLO model and free memory"""
    global _yolo_model, _last_yolo_used
    if _yolo_model is not None:
        try:
            import torch
            if hasattr(_yolo_model, 'model'):
                del _yolo_model.model
            del _yolo_model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error unloading YOLO model: {e}")
        finally:
            _yolo_model = None
            _last_yolo_used = 0


def load_models():
    """Load models if not already loaded"""
    global _yolo_model
    with _model_lock:
        if _yolo_model is None:
            _yolo_model = YOLO("./yolov8n_face_trained.pt")

def unload_models():
    """Unload all models and free memory"""
    with _model_lock:
        _unload_yolo_model()

def get_yolo_model():
    """Get YOLO model with lazy loading and memory management"""
    global _yolo_model, _last_yolo_used
    
    with _model_lock:
        # Check if model should be unloaded
        if _yolo_model is not None and _should_unload_model(_last_yolo_used):
            _unload_yolo_model()
        
        # Load model if not available
        if _yolo_model is None:
            _yolo_model = YOLO("./yolov8n_face_trained.pt")
        
        _last_yolo_used = time.time()
        return _yolo_model


def get_model_status():
    """Get current model loading status and memory usage"""
    return {
        "yolo_loaded": _yolo_model is not None,
        "memory_usage": _get_memory_usage(),
        "last_yolo_used": _last_yolo_used,
    }

def force_unload_models():
    """Force unload all models regardless of TTL"""
    with _model_lock:
        _unload_yolo_model()