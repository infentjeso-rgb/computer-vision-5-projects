# Road Pothole Detection Using Computer Vision

## 1. Project Overview

This project develops a computer vision system for detecting road potholes using a YOLO11n object detection model.

The system can:

- Detect multiple potholes in road images
- Draw bounding boxes around detected potholes
- Estimate detection confidence
- Count detected potholes
- Process webcam input in real time
- Display a warning when potholes are detected
- Save detection results for analysis

## 2. Problem Statement

Road potholes can create safety hazards for vehicles and road users. Manual road inspection is time-consuming and difficult to scale.

This project uses deep-learning-based object detection to automatically identify potholes from road images and camera input.

## 3. System Pipeline

```text
Road Image / Webcam
        ↓
Image Preprocessing
        ↓
YOLO11n Object Detection
        ↓
Pothole Detection
        ↓
Bounding Boxes + Confidence
        ↓
Pothole Count
        ↓
Warning System
