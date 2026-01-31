from ultralytics import YOLO
import cv2

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "runs/detect/train3/weights/best.pt"
model = YOLO(MODEL_PATH)

print("Model classes:", model.names)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONF_THRESHOLD = 0.2

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca kamera")
        break

    # (Optional) mirror webcam
    frame = cv2.flip(frame, 1)

    # =========================
    # INFERENCE
    # =========================
    results = model(
        frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    # =========================
    # DRAW RESULT
    # =========================
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Realtime Detection", annotated_frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
