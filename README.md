以下是一个基于上述内容的GitHub项目README示例，适用于描述该研究项目及其代码实现：

---

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

## Directory Structure
```
mouse_pose_estimation/
├── data/
│   ├── day_frames/
│   ├── night_frames/
│   └── labels/
├── models/
│   ├── yolov7_pose.py
│   ├── improved_yolov7_pose.py
│   └── svm_classifier.py
├── utils/
│   ├── cbam.py
│   ├── cfp_elan_w.py
│   ├── coordconv.py
│   └── feature_extraction.py
├── training/
│   ├── train.py
│   └── validate.py
├── evaluation/
│   ├── evaluate_keypoints.py
│   └── evaluate_behavior.py
├── README.md
└── LICENSE
```

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch 1.12.1
- CUDA 11.6
- NVIDIA GPU (Recommended: RTX 4090)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mouse_pose_estimation.git
   cd mouse_pose_estimation
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision numpy opencv-python scikit-learn
   ```

### Data Preparation
1. Place your day and night frame images in the `data/day_frames` and `data/night_frames` directories.
2. Label the key points using Labelme and convert the JSON files to TXT format in YOLO format.

### Training the Model
1. Train the improved YOLOv7-Pose model:
   ```bash
   python training/train.py --data_path data/ --model_path models/improved_yolov7_pose.pth
   ```

2. Validate the model:
   ```bash
   python training/validate.py --model_path models/improved_yolov7_pose.pth
   ```

### Evaluating Key Points
1. Evaluate key point detection performance:
   ```bash
   python evaluation/evaluate_keypoints.py --model_path models/improved_yolov7_pose.pth --data_path data/
   ```

### Behavior Classification
1. Extract features from key points:
   ```bash
   python utils/feature_extraction.py --data_path data/ --output_path features/
   ```

2. Train and evaluate the SVM classifier:
   ```bash
   python evaluation/evaluate_behavior.py --feature_path features/ --model_path models/svm_classifier.pkl
   ```

## Results
- **Key Point Detection**: The improved YOLOv7-Pose model achieves an F1 Score of 0.840 and accuracy of 84.1%.
- **Behavior Classification**: The model achieves an average classification accuracy of 90.22% for all videos, with improvements of 4.32% for day videos, 6.75% for night videos, and 5.86% overall compared to the original YOLOv7-Pose model.

## References
For more details about the methodology and results, please refer to the original publication:
- Yang, Y., Zhu, Y., Lu, W., et al. Enhanced Mouse Pose Estimation through Improved YOLOv7-Pose Model for Behavior Analysis. *Applied Sciences* 14(11), 4351 (2024). [DOI: 10.3390/app14114351](https://doi.org/10.3390/app14114351)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
This research is supported by grants from Hubei Hongshan Laboratory [grant numbers 2022hszd024] and Huazhong Agricultural University Scientific & Technological Self-innovation Foundation.

---

### Notes:
1. Replace `yourusername` with your actual GitHub username.
2. Ensure that the directory structure and file paths match those described in the README.
3. Add additional sections or details as needed based on your project requirements.
