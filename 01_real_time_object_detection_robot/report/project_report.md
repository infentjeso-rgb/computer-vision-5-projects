# Project Report
## Real-Time Object Detection for Robotic Applications

### 1. Introduction

This project implements a real-time computer vision system for detecting objects that may act as obstacles for a mobile robot. A laptop webcam is used as the camera input and YOLO11n is used for object detection.

The detected object's horizontal position is analysed to divide the scene into LEFT, CENTRE and RIGHT regions. Based on this position, the system generates a simulated robot navigation command.

### 2. Problem Statement

A mobile robot operating in an environment needs to identify surrounding objects before making movement decisions. Computer vision can provide the perception required to detect potential obstacles.

This project demonstrates a basic vision-based obstacle detection and decision-making pipeline.

### 3. Objectives

- Detect objects in real time using a webcam.
- Detect potential obstacles such as people, cars, buses, bicycles and motorcycles.
- Determine the horizontal position of detected obstacles.
- Divide the camera view into LEFT, CENTRE and RIGHT regions.
- Generate simulated navigation commands.
- Evaluate the object detection model using standard metrics.

### 4. Dataset

The project uses COCO128, a 128-image subset of the COCO dataset provided by Ultralytics.

COCO128 contains images with YOLO-format object detection annotations.

The YOLO11n model used in this project is pretrained on the COCO dataset. COCO128 was used for testing and evaluation rather than for training the model from scratch.

### 5. Methodology

The implemented pipeline is:

Webcam Input  
↓  
Image Frame  
↓  
YOLO11n Object Detection  
↓  
Confidence Filtering  
↓  
Obstacle Identification  
↓  
LEFT / CENTRE / RIGHT Position Detection  
↓  
Simulated Robot Decision

A confidence threshold of 0.50 is used.

The decision logic is:

| Obstacle Position | Simulated Command |
|---|---|
| No obstacle | MOVE |
| LEFT | AVOID RIGHT |
| CENTRE | STOP |
| RIGHT | AVOID LEFT |

### 6. Technologies Used

- Python
- YOLO11n
- Ultralytics
- OpenCV
- PyTorch
- CUDA
- NVIDIA GeForce RTX 3050 Laptop GPU
- Laptop Webcam

### 7. Implementation

The main implementation is contained in:

```text
src/object_detection.py
