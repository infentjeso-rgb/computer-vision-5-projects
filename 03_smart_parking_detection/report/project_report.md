mkdir -p 03_smart_parking_detection/report

cat > 03_smart_parking_detection/report/project_report.md <<'EOF'
# Project 3 — Automatic Smart Parking Space Detection and Occupancy Monitoring

## 1. Problem Statement

Monitoring multiple parking spaces manually is time-consuming. This project develops an automatic computer vision system that detects parking spaces from a parking-lot video and determines whether the detected spaces are empty or occupied.

## 2. Objective

The system aims to:

- Automatically detect parking spaces.
- Classify spaces as empty or occupied.
- Track repeated detections across video frames.
- Remove temporary detections.
- Generate a final parking-space map.
- Calculate parking occupancy statistics.

## 3. Input

The system uses a real parking-lot video:

`dataset/videos/parking_new.mp4`

Video properties:

- Resolution: 1026 × 578
- Frame rate: 30 FPS
- Total frames: 895

## 4. Methodology

```text
Parking Video
      ↓
YOLO Parking-Space Detection
      ↓
Empty / Occupied Classification
      ↓
Spatial Tracking Across Frames
      ↓
Stable Detection Filtering
      ↓
Majority-Vote Classification
      ↓
Parking Statistics
