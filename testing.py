from ultralytics import YOLO
import cv2

# Gunakan path absolut ke best.pt hasil training kamu
model = YOLO(r"C:/Users/rival\Documents/My code/ObjectDetectionWithDataset/runs\detect/train/weights/best.pt")

# Set conf=0.01 untuk melihat apakah model mendeteksi sesuatu meski tidak yakin
results = model.predict(
    source="test2.jpg", 
    conf=0.02, 
    save=True, 
    project="runs/detect", 
    name="debug_predict"
)

print("Cek folder runs/detect/debug_predict untuk hasilnya")