import cv2
import time
from ultralytics import YOLO

MODEL = "runs/detect/results/pothole_yolo11n/weights/best.pt"

model = YOLO(MODEL)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_time = time.time()

print("=" * 50)
print("LIVE ROAD POTHOLE DETECTION")
print("=" * 50)
print("Press Q to quit")
print("Press S to save screenshot")
print("=" * 50)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False
    )

    result = results[0]

    pothole_count = len(result.boxes)

    annotated_frame = result.plot()

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Detection information
    cv2.putText(
        annotated_frame,
        f"Potholes: {pothole_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if pothole_count > 0:

        cv2.putText(
            annotated_frame,
            "WARNING: POTHOLE DETECTED",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            annotated_frame,
            "ROAD CLEAR",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Road Pothole Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):

        filename = f"screenshots/live_pothole_{int(time.time())}.jpg"

        cv2.imwrite(filename, annotated_frame)

        print(f"Screenshot saved: {filename}")

cap.release()
cv2.destroyAllWindows()
