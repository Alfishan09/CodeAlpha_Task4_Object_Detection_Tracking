import streamlit as st
from ultralytics import YOLO
import cv2

# Page Configuration
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

# Sidebar
st.sidebar.title("Project Info")

st.sidebar.info(
    """
🎯 *Object Detection & Tracking System*

Developed by
Alfishan Khan

🤖 AI Intern | @CodeAlpha

Building real-time object detection and
tracking application using YOLOv8 and OpenCV.
"""
)

st.sidebar.subheader("Features")

st.sidebar.write("✅ Real-time Object Detection")
st.sidebar.write("✅ Multi-Object Tracking")
st.sidebar.write("✅ YOLOv8 Deep Learning Model")
st.sidebar.write("✅ Confidence Scoring")
st.sidebar.write("✅ Webcam Integration")

# Main Heading
st.markdown("""
<h1 style='text-align:center; color:white; font-size:46px; font-weight:800'>
🎯 Real-Time Object Detection & Tracking
</h1>
""", unsafe_allow_html=True)

# Powered by line
st.markdown("""
<h3 style='text-align:center;
           color:#00D4FF;
           font-style:italic;
           letter-spacing:1px;'>
Powered by YOLOv8
</h3>
""", unsafe_allow_html=True)

# Tagline
st.markdown("""
<h4 style='text-align:center; color:#CFCFCF;'>
Detect and track objects in real time using computer vision and deep learning.
</h4>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# Center Button
col1, col2, col3 = st.columns([2, 1, 2])

# Start Detection
with col2:
    if st.button("🚀 Start Detection"):

        st.success("🎥 Webcam Active | Press 'Q' to Exit")

        # Load YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Open webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Object Detection + Tracking
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.25
            )

            # Draw detections
            annotated_frame = results[0].plot()

            # Display output
            cv2.imshow(
                "CodeAlpha Object Detection & Tracking",
                annotated_frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

