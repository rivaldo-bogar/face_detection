from ultralytics import YOLO
import cv2

# Load 2 model
model_box = YOLO("runs/detect/train/weights/best.pt")   # boxstick
model_hand = YOLO("runs/detect/train3/weights/best.pt")  # hand gesture

cap = cv2.VideoCapture(1)
CONF = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Inference dua model
    res_box = model_box(frame, conf=CONF, verbose=False)[0]
    res_hand = model_hand(frame, conf=CONF, verbose=False)[0]

    # Draw boxstick (biru)
    for box in res_box.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, "Box Stick", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Draw hand (hijau)
    for box in res_hand.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, "Valdo", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Multi Model YOLO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
