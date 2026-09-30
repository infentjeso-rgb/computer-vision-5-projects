# Fruit Quality Inspection System

## 1. Introduction

The Fruit Quality Inspection System is a computer vision application that classifies fruit according to visible freshness quality.

The system uses a MobileNetV3 Small deep-learning model with transfer learning and supports real-time webcam inspection.

## 2. Objective

The objective is to classify fruit into two categories:

1. GOOD / FRESH
2. DEFECTIVE / ROTTEN

## 3. System Pipeline

Webcam -> Inspection Zone -> Image Preprocessing -> MobileNetV3 Small -> Prediction Smoothing -> Classification -> Confidence

## 4. Dataset

The project uses a Fresh vs Rotten Fruit Images dataset.

The dataset contains fruit images including apples, bananas and strawberries.

The images were organized into:

- Good / Fresh
- Defective / Rotten

## 5. Model

MobileNetV3 Small was selected as a lightweight deep-learning model.

Transfer learning was used with pretrained MobileNetV3 weights. The final classifier was modified for two classes.

Input size: 224 x 224

## 6. Training

Training configuration:

- Epochs: 8
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.0005
- GPU acceleration: CUDA

Best validation accuracy: 89.87%

## 7. Evaluation

The model was evaluated on 81 unseen test images.

Test accuracy: 98.77%

Macro F1-score: 98.65%

Weighted F1-score: 98.76%

Confusion Matrix:

                 Predicted
              Defective   Good
Actual Def.       28        1
Actual Good        0       52

Therefore, 80 of 81 test images were classified correctly.

## 8. Real-Time Inspection

The final application uses a webcam and a central inspection region.

The user places a fruit inside the green inspection box. The model processes the region and displays:

- GOOD / FRESH
- DEFECTIVE / ROTTEN
- Confidence percentage

Prediction smoothing is used to make the live output more stable.

## 9. Results

The system achieved 98.77% accuracy on the prepared test set.

The live camera application successfully performs real-time fruit quality classification using CUDA acceleration.

## 10. Applications

- Fruit quality inspection
- Automated sorting assistance
- Agricultural inspection
- Food quality monitoring
- Computer vision education and demonstration

## 11. Limitations

The model is limited to the freshness/quality categories represented in the training dataset.

It does not identify specific physical defect types and should not be considered a food-safety certification system.

## 12. Conclusion

The project demonstrates a complete computer vision workflow from dataset preparation and model training to quantitative evaluation and real-time webcam inference.

The MobileNetV3 Small model achieved 98.77% test accuracy on the prepared test set and was successfully integrated into a live webcam inspection system.
