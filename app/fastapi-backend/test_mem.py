import os
import psutil
import gc
import numpy as np
import cv2

from deepface import DeepFace
from services.model_loader import get_yolo_model, get_arcface_model

def print_ram(note=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[{note}] RAM Usage: {rss:.2f} MB")

def create_dummy_image():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.ellipse(img, (256, 256), (100, 140), 0, 0, 360, (255, 224, 189), -1)
    cv2.circle(img, (220, 220), 15, (0, 0, 0), -1)
    cv2.circle(img, (292, 220), 15, (0, 0, 0), -1)
    cv2.ellipse(img, (256, 310), (40, 20), 0, 0, 180, (0, 0, 0), 3)
    return img

def main():
    print_ram("Start")

    # Load YOLO model
    yolo = get_yolo_model()
    print_ram("After YOLO loaded")

    # Load ArcFace model
    arcface = get_arcface_model()
    print_ram("After ArcFace loaded")

    # Dummy image
    img = create_dummy_image()

    # YOLO inference
    results = yolo.predict(source=img, conf=0.25, verbose=False)
    boxes = results[0].boxes.xyxy
    print(f"YOLO detected {len(boxes)} faces")
    print_ram("After YOLO inference")

    if boxes:
        x1, y1, x2, y2 = map(int, boxes[0])
        face_crop = img[y1:y2, x1:x2]

        # ArcFace inference via DeepFace
        embedding = DeepFace.represent(
            face_crop,
            model_name="SFace",
            enforce_detection=False,
            align=True,
            model=arcface
        )
        print(f"ArcFace embedding length: {len(embedding[0]['embedding'])}")
        print_ram("After ArcFace inference")

    # Cleanup
    del yolo, arcface, img, boxes, results
    gc.collect()
    print_ram("After cleanup")

if __name__ == "__main__":
    main()
