import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vehicle Detection Dashboard", layout="wide")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detection", "Analytics","Model Evaluation"])

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("🚗 Vehicle Detection System")
    st.markdown("""
    ### Computer Vision Project
    
    This system performs:
    - Real-time vehicle detection
    - Vehicle classification
    - Video processing
    - Analytics visualization
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "YOLOv8")
    col2.metric("Classes", "5 Vehicles")
    col3.metric("Framework", "PyTorch")

# -----------------------------
# DETECTION PAGE
# -----------------------------
elif page == "Detection":
  
    st.title("🎥 Upload & Detect Vehicles")

    uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)

    
    if uploaded_file is not None:

        st.video(uploaded_file)

        if st.button("Run Detection"):

            file_extension = uploaded_file.name.split('.')[-1]

            # Create uploads folder if not exists
            os.makedirs("uploads", exist_ok=True)

            video_path = os.path.join("uploads", uploaded_file.name)

            with open(video_path, "wb") as f:
               f.write(uploaded_file.getbuffer())

            st.info("Processing video... Please wait!")
            

           results = model.predict(
                    source=video_path,
                    save=True,
                    conf=confidence,
                    imgsz=224,          # smaller image size (FASTER)
                    vid_stride=3,       # process every 3rd frame (VERY IMPORTANT)
                    device="cpu",
                    verbose=False
                 )
            
            save_dir = results[0].save_dir

            video_files = [
                f for f in os.listdir(save_dir)
                if f.endswith((".mp4", ".avi", ".mov"))
            ]

            if video_files:

                original_video = os.path.join(save_dir, video_files[0])
                final_video = os.path.join(save_dir, "browser_output.mp4")

                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
                    "-i", original_video,
                    "-vcodec", "libx264",
                    "-acodec", "aac",
                    final_video
                ]

                subprocess.run(ffmpeg_command)

                st.success("Detection Complete")

                with open(final_video, "rb") as f:
                    video_bytes = f.read()

                st.video(video_bytes)

                # -------- VEHICLE COUNTING --------
                class_counts = {}

                for r in results:
                   for box in r.boxes:
                     cls_id = int(box.cls)
                     cls_name = model.names[cls_id]

                     class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                st.session_state["counts"] = class_counts

            else:
                st.error("No output video found.")

# -----------------------------
# ANALYTICS PAGE
# -----------------------------
elif page == "Analytics":

    st.title("📊 Vehicle Analytics Dashboard")

    if "counts" in st.session_state:

        counts = st.session_state["counts"]

        total_vehicles = sum(counts.values())
        unique_classes = len(counts)
        most_common = max(counts, key=counts.get)

        # -------- METRICS --------
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Vehicles", total_vehicles)
        col2.metric("Vehicle Types Detected", unique_classes)
        col3.metric("Most Frequent", most_common)

        df = pd.DataFrame({
            "Vehicle Type": list(counts.keys()),
            "Count": list(counts.values())
        })

        st.subheader("Vehicle Count Table")
        st.dataframe(df)

        # -------- BAR CHART --------
        st.subheader("Bar Chart")
        st.bar_chart(df.set_index("Vehicle Type"))

        # -------- PIE CHART --------
        st.subheader("Vehicle Distribution (Pie Chart)")

        fig, ax = plt.subplots(figsize = (6,4))
        ax.pie(df["Count"], labels=df["Vehicle Type"], autopct="%1.1f%%")
        ax.set_title("Vehicle Distribution")
        st.pyplot(fig)

        # -------- DOWNLOAD CSV --------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Report as CSV",
            csv,
            "vehicle_report.csv",
            "text/csv"
        )

    else:
        st.warning("Run detection first to see analytics.")

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation Dashboard")

    if st.button("Run Validation"):

        st.info("Running validation on dataset... Please wait.")

        # Run validation
        metrics = model.val(data="data.yaml")

        st.success("Validation Complete")

        # =====================================
        # 📊 OVERALL METRICS
        # =====================================
        st.subheader("📊 Overall Performance")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("mAP@0.5", f"{metrics.box.map50:.3f}")
        col2.metric("mAP@0.5:0.95", f"{metrics.box.map:.3f}")
        col3.metric("Precision", f"{metrics.box.mp:.3f}")
        col4.metric("Recall", f"{metrics.box.mr:.3f}")

        # =====================================
        # 📋 PER-CLASS METRICS
        # =====================================
        st.subheader("📋 Per-Class Metrics")

        class_names = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

        df_metrics = pd.DataFrame({
            "Class": class_names,
            "Precision": metrics.box.p,
            "Recall": metrics.box.r,
            "mAP@0.5": metrics.box.ap50,
            "mAP@0.5:0.95": metrics.box.ap
        })

        # F1 Score (safe calculation)
        df_metrics["F1 Score"] = 2 * (
            df_metrics["Precision"] * df_metrics["Recall"]
        ) / (
            df_metrics["Precision"] + df_metrics["Recall"] + 1e-6
        )

        # Sort by mAP@0.5
        df_metrics_sorted = df_metrics.sort_values("mAP@0.5", ascending=False)

        st.dataframe(
            df_metrics_sorted.style
            .background_gradient(subset=["mAP@0.5"], cmap="Greens")
            .background_gradient(subset=["F1 Score"], cmap="Blues")
        )

        # =====================================
        # 🧠 BRIEF ANALYSIS
        # =====================================
        st.subheader("🧠 Brief Analysis")

        best_class = df_metrics.loc[df_metrics["mAP@0.5"].idxmax(), "Class"]
        worst_class = df_metrics.loc[df_metrics["mAP@0.5"].idxmin(), "Class"]

        st.write(f"""
        The model achieves an overall **mAP@0.5 of {metrics.box.map50:.2f}**, 
        indicating good object detection performance.

        The best performing class is **{best_class}**, while the weakest class is **{worst_class}**.

        A Precision of **{metrics.box.mp:.2f}** shows reliable predictions,
        while Recall of **{metrics.box.mr:.2f}** indicates moderate detection coverage.
        """)

        # =====================================
        # 🔎 CONFUSION MATRIX
        # =====================================
        st.subheader("🔎 Confusion Matrix")

        cm = metrics.confusion_matrix.matrix

        # Some YOLO versions add background class
        if cm.shape[0] > len(class_names):
            cm = cm[:len(class_names), :len(class_names)]

        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)

        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.title("Confusion Matrix")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center", color="black")

        st.pyplot(fig)

        # =====================================
        # 📊 mAP PER CLASS BAR CHART
        # =====================================
        st.subheader("📊 mAP@0.5 per Class")

        fig2, ax2 = plt.subplots()
        ax2.bar(df_metrics_sorted["Class"], df_metrics_sorted["mAP@0.5"])
        ax2.set_ylabel("mAP@0.5")
        ax2.set_title("mAP@0.5 per Class")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # =====================================
        # 📈 PR & F1 CURVES
        # =====================================
        import glob
        import os

        val_folders = sorted(glob.glob("runs/detect/val*"))

        if val_folders:
            latest_val = val_folders[-1]

            pr_curve = os.path.join(latest_val, "PR_curve.png")
            f1_curve = os.path.join(latest_val, "F1_curve.png")

            if os.path.exists(pr_curve):
                st.subheader("📈 Precision-Recall Curve")
                st.image(pr_curve)

            if os.path.exists(f1_curve):
                st.subheader("📈 F1 Score Curve")
                st.image(f1_curve)

        # =====================================
        # 📥 DOWNLOAD REPORT
        # =====================================
        st.subheader("📥 Download Evaluation Report")

        csv = df_metrics_sorted.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Evaluation Report (CSV)",
            csv,
            "evaluation_report.csv",
            "text/csv"
        )

        # =====================================
        # 🖼 SAMPLE PREDICTIONS
        # =====================================
        st.subheader("🖼 Sample Image  Predictions")

        predict_folders = sorted(glob.glob("runs/detect/val*"))

        if predict_folders:
            latest_predict = predict_folders[-1]

            image_files = glob.glob(os.path.join(latest_val, "*.jpg"))
            image_files += glob.glob(os.path.join(latest_val, "*.png"))

            if image_files:
                cols = st.columns(3)

                for i, img_path in enumerate(image_files[:6]):
                    cols[i % 3].image(img_path, use_container_width=True)
            else:
                st.warning("No annotated validation images found.")
        else:
            st.warning("Run detection first to generate prediction images.")

        st.subheader("Sample Vedio prediction")
        predict_folders = sorted(glob.glob("runs/detect/predict*"))

        if predict_folders:
            latest_predict = predict_folders[-1]

            
            # Show annotated video if exists
            video_files = [
                f for f in os.listdir(latest_predict)
                if f.endswith((".mp4", ".avi"))
            ]

            if video_files:
                st.subheader("🎬 Annotated Video")
                video_path = os.path.join(latest_predict, video_files[0])
                with open(video_path, "rb") as v:
                    st.video(v.read())

        else:
            st.warning("Run detection first to generate prediction images.")

