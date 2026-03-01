# 🚗 Vehicle Detection System using YOLOv8

A complete end-to-end Computer Vision project for **vehicle detection, analytics, and model evaluation** using **YOLOv8** and **Streamlit**.

---

## 📌 Project Overview

This project implements an intelligent vehicle detection system capable of:

- Real-time vehicle detection
- Multi-class vehicle classification
- Video processing with FFmpeg
- Interactive analytics dashboard
- Model performance evaluation
- Confusion matrix visualization

Built using:

- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **Streamlit**
- **OpenCV**
- **Matplotlib**
- **Pandas**
- **FFmpeg**

---

## 🚘 Classes Detected

The model detects 5 vehicle categories:

- Ambulance
- Bus
- Car
- Motorcycle
- Truck

---

## 📊 Application Features

### 🔹 1. Detection Page
- Upload traffic video
- Adjustable confidence threshold slider
- Browser-compatible processed output
- Automatic vehicle counting
- Annotated video output

---

### 🔹 2. Analytics Dashboard
- Total vehicle count
- Most frequent vehicle type
- Interactive bar chart
- Pie chart distribution
- Downloadable CSV report

---

### 🔹 3. Model Evaluation Dashboard
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1 Score
- Per-class performance table
- Confusion matrix visualization
- mAP per class bar chart
- PR Curve & F1 Curve
- Downloadable evaluation report

---

## 🗂 Project Structure
Computer_Vision_Project/
│
├── app.py # Streamlit application
├── train.py # Model training script
├── inference.py # Inference script (image/video/webcam)
├── best.pt # Trained model weights
├── data.yaml # Dataset configuration
├── requirements.txt # Dependencies
├── README.md
│
├── VehiclesDetectionDataset/
│ ├── train/
│ ├── valid/
│ └── test/
│
├── uploads/ # Uploaded videos
└── runs/ # YOLO output folders


---

## ⚙ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd Computer_Vision_Project

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

4️⃣ Install FFmpeg (Required for Video Processing)

Download from:

👉 https://www.gyan.dev/ffmpeg/builds/

Add FFmpeg to your system PATH.

Verify installation:
ffmpeg -version

Run the Streamlit Application
streamlit run app.py

Model Training (Optional)

To retrain the model:

python train.py

Trained weights will be saved in:

runs/detect/train*/weights/best.pt
📈 Evaluation Metrics Explained
Metric	Description
mAP@0.5	Detection accuracy at IoU threshold 0.5
mAP@0.5:0.95	Stricter averaged detection accuracy
Precision	Correct detections among predicted detections
Recall	Correct detections among actual objects
F1 Score	Harmonic mean of Precision and Recall
📦 Project Deliverables

✅ Training Script

✅ Inference Script (Image + Video)

✅ Dataset YAML

✅ Trained Weights (best.pt)

✅ Evaluation Report

✅ Confusion Matrix

✅ Annotated Images

✅ Annotated Video

✅ Interactive Streamlit Dashboard

🚀 Future Improvements

Real-time webcam detection

Vehicle speed estimation

Multi-object tracking

Deployment on Streamlit Cloud

Model comparison (YOLOv8n vs YOLOv8s)

REST API integration

👩‍💻 Author

Priyanka Kumawat
Computer Vision & Data Science Enthusiast

⭐ If You Like This Project

Give it a ⭐ on GitHub!