cat > README.md <<'EOF'
# Project 3 — Automatic Smart Parking Space Detection and Occupancy Monitoring

## 1. Problem Statement

Parking areas contain multiple parking spaces that may be empty or occupied. Manually monitoring every parking space is time-consuming.

This project develops an automatic computer vision system that detects parking spaces from a parking-lot video and classifies them as empty or occupied.

## 2. Objective

The system aims to:

- Automatically detect parking spaces.
- Classify each detected space as empty or occupied.
- Track repeated detections across video frames.
- Remove temporary detections.
- Generate a final parking-space map.
- Calculate total, empty, occupied and occupancy statistics.

## 3. Input

Input video:

`dataset/videos/parking_new.mp4`

Video properties:

- Resolution: 1026 × 578
- FPS: 30
- Frames: 895

## 4. Methodology

```text
Parking Video
      ↓
YOLO Parking-Space Detection
      ↓
Empty / Occupied Classification
      ↓
Temporal Space Tracking
      ↓
Duplicate / Temporary Detection Filtering
      ↓
Stable Parking Spaces
      ↓
Majority-Vote Classification
      ↓
Parking Statistics
