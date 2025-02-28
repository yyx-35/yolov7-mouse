# Enhanced Mouse Pose Estimation through Improved YOLOv7-Pose Model

## Project Overview
This repository contains the implementation of an improved YOLOv7-Pose model for accurate mouse key point detection and behavior analysis. The model incorporates three key enhancements to address the challenge of low accuracy in key point recognition, especially in blurry images caused by rapid mouse movement. These enhancements include:
1. **Convolution Block Attention Module (CBAM)**: Enhances the model's attention towards mouse key points.
2. **Centralized Feature Pyramid (CFP) in ELAN-W**: Improves the model's capacity to capture global long-range dependencies.
3. **CoordConv Layers**: Enhances detection accuracy by incorporating spatial perception.

The improved model achieves an F1 Score of 0.840 and accuracy of 84.1%, outperforming the original YOLOv7-Pose model by 9.64% and 9.66% respectively. It is applied to extract key points from day and night frame images, and the derived features are used to classify seven mouse behaviors using a Support Vector Machine (SVM).

## Key Features
- **Enhanced Key Point Detection**: Improved YOLOv7-Pose model for accurate detection of mouse key points.
- **Behavior Classification**: Utilizes key point features to classify mouse behaviors such as rearing, resting, sniffing, climbing, grooming, walking, and eating.
- **Robustness in Low-Quality Images**: Enhanced performance in blurry and low-light conditions.

## Results
- **Key Point Detection**: The improved YOLOv7-Pose model achieves an F1 Score of 0.840 and accuracy of 84.1%.
- **Behavior Classification**: The model achieves an average classification accuracy of 90.22% for all videos, with improvements of 4.32% for day videos, 6.75% for night videos, and 5.86% overall compared to the original YOLOv7-Pose model.

## Acknowledgments
This research is supported by grants from Hubei Hongshan Laboratory [grant numbers 2022hszd024] and Huazhong Agricultural University Scientific & Technological Self-innovation Foundation.
