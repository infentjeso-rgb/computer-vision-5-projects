from ultralytics import YOLO

MODEL = "runs/detect/results/pothole_yolo11n/weights/best.pt"
DATA = "dataset/yolo/data.yaml"

model = YOLO(MODEL)

metrics = model.val(
    data=DATA,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    plots=True,
    project="evaluation",
    name="test_results",
    exist_ok=True
)

print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print("=" * 60)
