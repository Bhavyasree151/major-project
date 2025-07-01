# major-project
# 🛠️ Object Detection & Distance Estimation for Indoor Autonomous Vehicles

This project focuses on building a real-time system that combines **object detection** and **depth estimation** to help indoor autonomous vehicles navigate safely.

## 🚀 Overview

We implemented a pipeline using multiple object detection models and a depth estimation model to:
- Detect obstacles in real-time
- Estimate the distance to each object from the camera

Our system uses a **hybrid model (YOLOv5 + YOLOv8)** for improved detection accuracy and **MiDaS** for depth prediction.

---

## 📂 Dataset

We used:
- 📦 **KITTI dataset** (limited to images suitable for indoor object detection)
- 🏠 **Custom indoor dataset** with 500+ manually labeled images in YOLO format

Data is split into:
/dataset/
├── train/
├── val/
└── test/

Each set includes:
- Images (`.jpg`)
- Labels (`.txt` in YOLO format)

---

### 🔹 Object Detection
- **YOLOv5** (variants: s, m, n, l)
- **YOLOv8**
- **SSD (Single Shot Multibox Detector)**

### 🔹 Hybrid Approach
- Combines predictions from **YOLOv5 and YOLOv8** to increase detection accuracy and robustness

### 🔹 Distance Estimation
- **MiDaS** (state-of-the-art monocular depth estimation model)

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Ultralytics YOLOv5 & YOLOv8**
- **TorchVision (for SSD)**
- **OpenCV**
- **Streamlit** (for UI demo)

---

## 📊 Evaluation Metrics

- mAP (mean Average Precision)
- Precision / Recall
- IoU (Intersection over Union)
- Distance estimation error (depth accuracy)

---

## 💻 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/object-detection-distance-estimation.git
   cd object-detection-distance-estimation
   Run the integrated Streamlit app: streamlit run app.py
