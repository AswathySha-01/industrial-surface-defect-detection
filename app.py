
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Industrial Surface Defect Detection")
st.write(
    "AI-powered system for detecting defects on metal surfaces."
)

# Model path
MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

if not MODEL_PATH.is_file():
    st.error(f"Model weights not found: {MODEL_PATH}")
    st.stop()


# Load model
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model weights: {exc}")
    st.stop()

st.divider()

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload multiple metal surface images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Detect defects in every uploaded image
if uploaded_files:
    if st.button("🔍 Detect Defects in All Images", type="primary"):

        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"Image: {uploaded_file.name}")

            image_bytes = np.frombuffer(
                uploaded_file.getvalue(),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                image_bytes,
                cv2.IMREAD_COLOR
            )

            if image is None:
                st.error("Unable to read this image.")
                continue

            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                results = model.predict(
                    source=image,
                    conf=0.35,
                    iou=0.45,
                    max_det=20
                )

            result = results[0]

            original_rgb = cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB
            )

            detected_rgb = cv2.cvtColor(
                result.plot(), cv2.COLOR_BGR2RGB
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("Original Image")
                st.image(
                    original_rgb,
                    use_container_width=True
                )

            with col2:
                st.write("Detection Result")
                st.image(
                    detected_rgb,
                    use_container_width=True
                )

            boxes = result.boxes
            count = len(boxes)

            class_names = [
                model.names[int(box.cls[0].item())]
                for box in boxes
            ]

            unique_classes = sorted(set(class_names))

            m1, m2 = st.columns(2)
            m1.metric("Total Detections", count)
            m2.metric("Defect Types", len(unique_classes))

            if count > 0:
                st.write("**Detected Defect Details**")

                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0].item())

                    st.write(
                        f"{i + 1}. **{class_name}** — "
                        f"Confidence: {confidence:.2%}"
                    )
            else:
                st.info("No defects detected in this image.")

            # Download this image's detection result
            success, encoded_image = cv2.imencode(
                ".jpg",
                cv2.cvtColor(detected_rgb, cv2.COLOR_RGB2BGR)
            )

            if success:
                st.download_button(
                    f"⬇️ Download {uploaded_file.name} Result",
                    data=encoded_image.tobytes(),
                    file_name=f"detected_{Path(uploaded_file.name).stem}.jpg",
                    mime="image/jpeg",
                    key=f"download_{uploaded_file.name}"
                )

else:
    st.info("Please upload one or more images to get started.")
