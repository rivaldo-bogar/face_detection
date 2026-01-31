from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=640,
        batch=3,
        device=0,
        workers=0   # <- PENTING
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
