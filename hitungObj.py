from ultralytics import YOLO
import cv2

# =========================
# BEEP (WINDOWS)
# =========================
try:
    import winsound
    def beep():
        winsound.Beep(1000, 200)  # freq, duration
except:
    def beep():
        print("BEEP")  # fallback

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

print("Model classes:", model.names)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONF_THRESHOLD = 0.2

# =========================
# COUNTER VARIABLE
# =========================
count = 0
object_present = False  # flag object sedang terlihat

TARGET_CLASS_ID = 0  # class boxstick

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca kamera")
        break

    frame = cv2.flip(frame, 1)

    # =========================
    # INFERENCE
    # =========================
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    detections = results[0].boxes
    current_detected = False

    # =========================
    # CHECK OBJECT CLASS 0
    # =========================
    if detections is not None:
        for box in detections:
            cls_id = int(box.cls[0])
            if cls_id == TARGET_CLASS_ID:
                current_detected = True
                break

    # =========================
    # COUNT LOGIC
    # =========================
    # kondisi: sebelumnya TIDAK ada → sekarang ADA
    if current_detected and not object_present:
        count += 1
        beep()
        object_present = True

    # reset jika object keluar frame
    if not current_detected:
        object_present = False

    # =========================
    # DRAW RESULT
    # =========================
    annotated_frame = results[0].plot()

    # =========================
    # DISPLAY COUNT
    # =========================
    cv2.putText(
        annotated_frame,
        f"COUNT: {count}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 0),
        3
    )

    cv2.imshow("Object count by valdo", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
