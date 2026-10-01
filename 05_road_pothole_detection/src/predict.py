from ultralytics import YOLO
import sys
from pathlib import Path

MODEL = "runs/detect/results/pothole_yolo11n/weights/best.pt"

if len(sys.argv) < 2:
    print("Usage:")
    print("python src/predict.py <image_path>")
    print()
    print("Example:")
    print("python src/predict.py screenshots/road.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not Path(image_path).exists():
    print(f"Error: Image not found: {image_path}")
    sys.exit(1)

model = YOLO(MODEL)

results = model.predict(
    source=image_path,
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    project="results",
    name="predictions",
    exist_ok=True
)

result = results[0]

print("=" * 50)
print("ROAD POTHOLE DETECTION")
print("=" * 50)

pothole_count = len(result.boxes)

print(f"Potholes detected: {pothole_count}")

if pothole_count > 0:
    print("\nDetections:")

    for i, box in enumerate(result.boxes, 1):
        confidence = float(box.conf[0])
        print(f"  Pothole {i}: {confidence * 100:.2f}% confidence")
else:
    print("No potholes detected.")

print("=" * 50)
print("Result saved in:")
print("results/predictions/")
