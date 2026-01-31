from ultralytics import YOLO
import cv2

# Gunakan path train yang sesuai dengan grafik tersebut (misal train7)
model = YOLO('runs/detect/train7/weights/best.pt') 

# Ambil SALAH SATU foto dari folder val Anda
img_path = 'dataset/images/val/IMG_20260101_212708.jpg'

results = model(img_path)
for r in results:
    # Jika di sini muncul kotak, berarti model HANYA bisa deteksi foto itu saja (overfit)
    r.show()