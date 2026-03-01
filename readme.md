# Vehicle Detection System using YOLOv8

A complete Computer Vision project for vehicle detection, analytics, and model evaluation using YOLOv8 and Streamlit.

---

##  Project Overview

This project performs:

- Real-time vehicle detection
- Vehicle classification (5 classes)
- Video processing with FFmpeg
- Analytics dashboard
- Model evaluation metrics
- Confusion matrix visualization

Built using:
- YOLOv8 (Ultralytics)
- Streamlit
- PyTorch
- OpenCV
- Matplotlib
- Pandas

---

## 🚘 Classes Detected

- Ambulance
- Bus
- Car
- Motorcycle
- Truck

---

## 📊 Features

### 🔹 Detection Page
- Upload traffic video
- Adjustable confidence threshold
- Browser-compatible processed video
- Vehicle counting

### 🔹 Analytics Dashboard
- Total vehicle count
- Most frequent vehicle
- Bar chart visualization
- Pie chart distribution
- Download CSV report

### 🔹 Model Evaluation
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1 Score
- Per-class metrics
- Confusion matrix
- mAP per class graph

---

## Project Structure
Computer_Vision_Project/
│
├── app.py
├── best.pt
├── data.yaml
├── requirements.txt
├── README.md
│
├── VehiclesDetectionDataset/
│ ├── train/
│ ├── valid/
│ ├── test/
│
└── uploads/


---

## ⚙ Installation

### Clone the repository
git clone <your-repo-link>
cd Computer_Vision_Project 


###  Create virtual environment

python -m venv venv
venv\Scripts\activate 


###  Install dependencies

pip install -r requirements.txt 


###  Install FFmpeg (Required)

Download from:
https://www.gyan.dev/ffmpeg/builds/

Add to system PATH.

Verify installation: ffmpeg -version

##  Run the App
streamlit run app.py 


---

##  Evaluation Metrics Explained

- **mAP@0.5** → Detection accuracy at IoU threshold 0.5  
- **mAP@0.5:0.95** → Stricter average detection accuracy  
- **Precision** → Correct detections among predicted detections  
- **Recall** → Correct detections among actual objects  
- **F1 Score** → Balance between Precision and Recall  

---

##  Technologies Used

- YOLOv8
- PyTorch
- Streamlit
- OpenCV
- FFmpeg
- Matplotlib
- Pandas

---

##  Future Improvements

- Real-time webcam detection
- Speed estimation
- Vehicle tracking
- Deployment on Streamlit Cloud
- Model comparison

---

##  Author

Priyanka Kumawat  
Computer Vision & Data Science Enthusiast 



