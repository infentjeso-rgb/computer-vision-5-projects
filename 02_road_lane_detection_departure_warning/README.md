# Road Lane Detection and Departure Warning

## 1. Problem Statement

Lane departure is a common road-safety problem in which a vehicle unintentionally moves away from its lane. This project develops a computer-vision-based system that detects road lane boundaries and provides a warning when the estimated vehicle position moves away from the lane centre.

## 2. Objective

The objectives are:

- Detect left and right road lane boundaries.
- Estimate the centre of the detected lane.
- Estimate the vehicle/image centre.
- Calculate the lateral offset between them.
- Generate lane departure warnings.
- Evaluate the system using TuSimple ground-truth annotations.

## 3. Dataset

**Dataset:** TuSimple Lane Detection Benchmark

A subset of 150 images was selected for this project from the TuSimple dataset.

Image resolution:

- 1280 × 720 pixels

The lane annotations were obtained from:

`label_data_0313.json`

Dataset source:

https://www.kaggle.com/datasets/manideep1108/tusimple

The dataset is used according to its stated attribution requirements.

## 4. Methodology

The system uses a classical computer vision pipeline:

```text
Input Road Image
       ↓
Grayscale Conversion
       ↓
Gaussian Blur
       ↓
Canny Edge Detection
       ↓
White / Yellow Lane Filtering
       ↓
Region of Interest
       ↓
Hough Line Transform
       ↓
Left / Right Lane Selection
       ↓
Lane Centre Estimation
       ↓
Vehicle Centre Comparison
       ↓
Departure Warning

cat evaluation/metrics.txt
