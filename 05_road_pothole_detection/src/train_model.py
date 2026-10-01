from ultralytics import YOLO
import torch

print("=" * 60)
print("ROAD POTHOLE DETECTION - YOLO11n TRAINING")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = YOLO("yolo11n.pt")

results = model.train(
    data="dataset/yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    project="results",
    name="pothole_yolo11n",
    patience=10,
    pretrained=True,
    exist_ok=True
)

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Best model:")
print("results/pothole_yolo11n/weights/best.pt")
