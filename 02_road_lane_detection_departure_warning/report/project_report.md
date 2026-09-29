# Project Report — Road Lane Detection and Departure Warning

## Abstract

This project implements a classical computer vision system for detecting road lanes and generating lane departure warnings. The implementation uses OpenCV-based preprocessing, Canny edge detection, colour filtering, region-of-interest extraction, and the Hough Line Transform.

A subset of 150 images from the TuSimple Lane Detection Benchmark was processed. Ground-truth lane annotations were used to evaluate the estimated lane centre.

## Problem Statement

Unintended movement away from a road lane can create a safety risk. A vision-based system can monitor lane boundaries and provide a warning when the estimated vehicle position deviates from the lane centre.

## Objectives

1. Detect road lane boundaries.
2. Estimate the lane centre.
3. Compare the lane centre with the vehicle/image centre.
4. Generate left/right departure warnings.
5. Quantitatively evaluate the system.

## Dataset

The project uses 150 selected images from the TuSimple Lane Detection Benchmark.

Image resolution: 1280 × 720 pixels.

Lane annotations are provided through `label_data_0313.json`.

## Methodology

The implemented pipeline is:

Input Image
↓
Grayscale Conversion
↓
Gaussian Blur
↓
Canny Edge Detection
↓
White/Yellow Lane Filtering
↓
Region of Interest
↓
Hough Line Transform
↓
Left/Right Lane Selection
↓
Lane Centre Estimation
↓
Vehicle Centre Comparison
↓
Departure Warning

## Implementation

Python and OpenCV are used for the implementation.

Candidate road-line segments are detected using the Hough Line Transform. The detected lines are classified according to their slope and position relative to the image centre.

The selected left and right lane boundaries are used to estimate the lane centre.

## Warning States

The system generates four states:

- LANE SAFE
- LANE DEPARTURE - LEFT
- LANE DEPARTURE - RIGHT
- LANE DETECTION INCOMPLETE

## Evaluation

The detector was evaluated against the TuSimple annotations.

| Metric | Result |
|---|---:|
| Images processed | 150 |
| Ground-truth matched | 142 |
| Successful detections | 123 |
| Incomplete detections | 19 |
| Detection completeness | 86.62% |
| Mean Absolute Error | 65.91 pixels |
| Median error | 39.50 pixels |
| Accuracy within 50 pixels | 55.28% |

## Predicted Status

| Status | Count |
|---|---:|
| Lane Safe | 80 |
| Departure Left | 12 |
| Departure Right | 31 |

## Results

The implementation successfully processes the selected dataset and produces visual lane overlays and lane departure warning outputs.

The quantitative evaluation indicates that the detector can identify lane structures in many of the selected images, while lane-centre estimation remains sensitive to complex road markings and environmental conditions.

## Limitations

The classical Hough-based approach can be affected by:

- Shadows
- Faded lane markings
- Multiple road lines
- Vehicles
- Curved lanes
- Poor illumination
- Missing lane boundaries

## Future Improvements

Future development can include:

- Bird's-eye-view transformation
- Polynomial lane fitting
- Temporal filtering
- Deep-learning-based lane segmentation
- Improved departure classification
- Real-time camera deployment
- Vehicle or robot integration

## Conclusion

This project demonstrates a complete classical computer vision pipeline for road lane detection and lane departure warning using OpenCV. The system processes road images, estimates lane position, generates warning states, and provides quantitative evaluation using TuSimple ground-truth lane annotations.
