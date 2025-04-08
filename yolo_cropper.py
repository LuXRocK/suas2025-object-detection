import cv2
from ultralytics import YOLO
import os

# Wczytaj model YOLO
model = YOLO('yolov8s.pt')

# Przechowuj ID już zapisanych obiektów
tracked_ids = set()

video_path = "DJI_0064.MP4"

# Uruchom detekcję na kamerze
for result in model.track(source=4, show=True, stream=True, conf = 0.8):
    frame = result.orig_img  # Pobierz aktualną klatkę
    
    if result.boxes.id is not None:
        for box, obj_id in zip(result.boxes.xyxy, result.boxes.id):
            obj_id = int(obj_id)
            if obj_id not in tracked_ids:  # Jeśli ID jest nowe
                tracked_ids.add(obj_id)
                x1, y1, x2, y2 = map(int, box)
                cropped_img = frame[y1:y2, x1:x2]
                
                # Tworzenie nazwy pliku
                cropped_path = f"cropped_images/tracked_object_{obj_id}.png"
                cv2.imwrite(cropped_path, cropped_img)
                print(f"Zapisano: {cropped_path}")

cv2.waitKey(0)
