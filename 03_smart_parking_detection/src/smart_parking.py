from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

MODEL = "models/parking_space_yolov8n.pt"
VIDEO = "dataset/videos/parking_new.mp4"

OUT_VIDEO = "results/smart_parking_output.mp4"
OUT_IMAGE = "results/smart_parking_final.jpg"
OUT_STATS = "evaluation/parking_statistics.txt"

CONF = 0.25
SAMPLE_STEP = 10
MATCH_DISTANCE = 55
MIN_SEEN = 8

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height}")
print(f"FPS: {fps:.2f}")
print(f"Frames: {total_frames}")

# Each slot:
# [x, y, width, height, labels, seen]
slots = []

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % SAMPLE_STEP == 0:

        result = model.predict(
            frame,
            conf=CONF,
            verbose=False
        )[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1

            # Ignore detections in sky / extreme borders
            if cy < height * 0.55:
                continue

            if cx < 30 or cx > width - 30:
                continue

            # Find an existing parking-space track
            best_index = -1
            best_distance = MATCH_DISTANCE

            for i, slot in enumerate(slots):
                sx, sy = slot[0], slot[1]
                distance = np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)

                if distance < best_distance:
                    best_distance = distance
                    best_index = i

            if best_index >= 0:
                slot = slots[best_index]

                # Smooth position
                slot[0] = 0.7 * slot[0] + 0.3 * cx
                slot[1] = 0.7 * slot[1] + 0.3 * cy
                slot[2] = 0.7 * slot[2] + 0.3 * bw
                slot[3] = 0.7 * slot[3] + 0.3 * bh

                slot[4].append(label)
                slot[5] += 1

            else:
                slots.append([
                    cx, cy, bw, bh,
                    [label],
                    1
                ])

    frame_index += 1

cap.release()

# Keep only persistent parking spaces
stable_slots = [
    slot for slot in slots
    if slot[5] >= MIN_SEEN
]

print(f"Raw tracks: {len(slots)}")
print(f"Stable parking spaces: {len(stable_slots)}")

# Determine final class by majority vote
final_slots = []

for slot in stable_slots:
    labels = slot[4]
    majority = Counter(labels).most_common(1)[0][0]

    final_slots.append({
        "cx": slot[0],
        "cy": slot[1],
        "w": slot[2],
        "h": slot[3],
        "label": majority,
        "seen": slot[5]
    })

# Sort spatially for reproducible IDs
final_slots.sort(key=lambda s: (s["cy"], s["cx"]))

for i, slot in enumerate(final_slots, start=1):
    slot["id"] = i

occupied = sum(
    1 for s in final_slots
    if s["label"] == "ocupied"
)

empty = sum(
    1 for s in final_slots
    if s["label"] == "empty"
)

total = len(final_slots)

occupancy = (occupied / total * 100) if total else 0

print()
print("===================================")
print("       SMART PARKING RESULTS")
print("===================================")
print(f"TOTAL    : {total}")
print(f"EMPTY    : {empty}")
print(f"OCCUPIED : {occupied}")
print(f"OCCUPANCY: {occupancy:.2f}%")
print("===================================")

# Save statistics
import os
os.makedirs("results", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

with open(OUT_STATS, "w") as f:
    f.write("SMART PARKING EVALUATION\n")
    f.write("========================\n")
    f.write(f"Input video: {VIDEO}\n")
    f.write(f"Video frames: {total_frames}\n")
    f.write(f"FPS: {fps:.2f}\n")
    f.write(f"Stable parking spaces: {total}\n")
    f.write(f"Empty spaces: {empty}\n")
    f.write(f"Occupied spaces: {occupied}\n")
    f.write(f"Occupancy: {occupancy:.2f}%\n")
    f.write(f"Confidence threshold: {CONF}\n")
    f.write(f"Sampling interval: {SAMPLE_STEP} frames\n")

# Generate final annotated image
cap = cv2.VideoCapture(VIDEO)

# Take a representative frame around the middle
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

ret, frame = cap.read()
cap.release()

if ret:

    for slot in final_slots:

        cx = int(slot["cx"])
        cy = int(slot["cy"])
        bw = int(slot["w"])
        bh = int(slot["h"])

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(width - 1, int(cx + bw / 2))
        y2 = min(height - 1, int(cy + bh / 2))

        if slot["label"] == "empty":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"S{slot['id']} {slot['label']}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    panel = np.zeros((100, width, 3), dtype=np.uint8)

    cv2.putText(
        panel,
        f"TOTAL: {total}",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        panel,
        f"EMPTY: {empty}",
        (220, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        panel,
        f"OCCUPIED: {occupied}",
        (420, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        panel,
        f"OCCUPANCY: {occupancy:.1f}%",
        (720, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    final_image = np.vstack((panel, frame))
    cv2.imwrite(OUT_IMAGE, final_image)

# Generate complete output video
cap = cv2.VideoCapture(VIDEO)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUT_VIDEO,
    fourcc,
    fps,
    (width, height + 100)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    for slot in final_slots:

        cx = int(slot["cx"])
        cy = int(slot["cy"])
        bw = int(slot["w"])
        bh = int(slot["h"])

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(width - 1, int(cx + bw / 2))
        y2 = min(height - 1, int(cy + bh / 2))

        color = (
            (0, 255, 0)
            if slot["label"] == "empty"
            else (0, 0, 255)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"S{slot['id']}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    panel = np.zeros((100, width, 3), dtype=np.uint8)

    cv2.putText(
        panel,
        f"SMART PARKING | TOTAL: {total}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        panel,
        f"FREE: {empty}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        panel,
        f"OCCUPIED: {occupied}",
        (220, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        panel,
        f"OCCUPANCY: {occupancy:.1f}%",
        (500, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    output = np.vstack((panel, frame))
    writer.write(output)

cap.release()
writer.release()

print()
print("FINAL FILES:")
print(OUT_IMAGE)
print(OUT_VIDEO)
print(OUT_STATS)
