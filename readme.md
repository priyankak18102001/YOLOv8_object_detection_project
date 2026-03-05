# 🚗 Vehicle Detection & Analytics Dashboard (YOLOv8 + Streamlit)

An interactive **Computer Vision dashboard** that detects vehicles from uploaded videos using **YOLOv8** and provides **analytics, visualizations, and model evaluation**.

This project demonstrates how **deep learning models can be deployed as real-time analytics applications** using Streamlit.

---

# 📌 Project Overview

The system performs:

* 🚗 Vehicle detection from uploaded videos
* 🎯 Object classification using YOLOv8
* 📊 Interactive analytics dashboard
* 📈 Model evaluation metrics
* 🎥 Annotated video output
* 📥 Downloadable reports

This project simulates a **Smart Traffic Monitoring System** used in intelligent transportation and smart city solutions.

---

# 🧠 Model

* Model: **YOLOv8**
* Framework: **PyTorch**
* Classes detected:

  * Ambulance
  * Bus
  * Car
  * Motorcycle
  * Truck

---

# ⚙️ Features

### 🎥 Vehicle Detection

* Upload a traffic video
* Detect vehicles using YOLOv8
* Display annotated video with bounding boxes
* Download annotated video

### 📊 Analytics Dashboard

* Vehicle count summary
* Interactive bar charts
* Vehicle distribution pie chart
* Detection trends
* Traffic insights
* CSV report download

### 📈 Model Evaluation

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall
* Confusion Matrix
* Precision–Recall Curve
* Per-class performance metrics

---

# 🖥️ Application Interface

## Home Page

Overview of the system and project information.

## Detection Page

Upload videos and run the vehicle detection model.

## Analytics Dashboard

Interactive charts showing vehicle distribution and traffic insights.

## Model Evaluation

Displays validation metrics and model performance.

---

# 📊 Example Analytics

The dashboard generates insights such as:

* Total vehicles detected
* Most frequent vehicle type
* Vehicle distribution charts
* Traffic behavior insights

---

# 🗂️ Project Structure

```
YOLOv8_object_detection_project
│
├── app.py
├── train.py
├── inference.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── best.pt
├── yolov8n.pt
│
├── VehiclesDetectionDataset
│   ├── train
│   ├── valid
│   ├── test
│   └── dataset.yaml
│
├── uploads
├── runs
└── README.md
```

---

# 🚀 Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/priyankak18102001/YOLOv8_object_detection_project.git
```

### 2️⃣ Navigate to the project folder

```
cd YOLOv8_object_detection_project
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

# 📦 Requirements

Main libraries used:

* streamlit
* ultralytics
* torch
* opencv-python
* pandas
* matplotlib
* plotly
* numpy

---

# 📊 Model Performance

Example evaluation metrics:

| Metric       | Score |
| ------------ | ----- |
| mAP@0.5      | 0.629 |
| mAP@0.5:0.95 | 0.485 |
| Precision    | 0.737 |
| Recall       | 0.548 |

---

# 💡 Applications

This system can be used for:

* Smart traffic monitoring
* Vehicle counting systems
* Smart city analytics
* Road traffic analysis
* Surveillance systems

---

# 🧑‍💻 Author

**Priyanka Kumawat**

Data Science & Computer Vision Enthusiast

GitHub:
https://github.com/priyankak18102001

LinkedIn:
https://www.linkedin.com/in/priyanka-kumawat-7177092a3

---

# ⭐ Future Improvements

* Real-time CCTV vehicle detection
* Vehicle speed estimation
* Traffic congestion analysis
* Live camera integration
* Vehicle heatmap visualization

---

# 📜 License

This project is for educational and research purposes.

t
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
