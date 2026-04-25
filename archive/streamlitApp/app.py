import streamlit as st
import cv2
import numpy as np
import pandas as pd

from image_processing import process_leaf_image
from model_inference import predict_leaf_class
from gradcam import make_gradcam_heatmap, overlay_gradcam


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="QBR Leaf Analyzer",
    page_icon="🌿",
    layout="wide"
)


# =========================
# Load external CSS
# =========================
def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("styles.css")


# =========================
# Helper functions
# =========================
def resize_for_display(image, width=420):
    h, w = image.shape[:2]
    new_height = int((width / w) * h)
    return cv2.resize(image, (width, new_height))


def friendly_class_name(class_name):
    return class_name.replace("Tomato_", "").replace("_", " ").title()


# =========================
# Header
# =========================
st.markdown(
    """
    <div class="hero-section">
        <div class="app-title">🌿 Quantitative Botanical Reasoning (QBR) System</div>
        <div class="app-subtitle">
            Upload a tomato leaf image to classify disease, estimate pixel-based severity,
            and visualize model attention using Grad-CAM.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Upload
# =========================
st.markdown('<div class="section-title">Upload Leaf Image</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Main app
# =========================
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read the uploaded image.")
    else:
        # -------------------------
        # Image Processing
        # -------------------------
        results = process_leaf_image(img)

        # -------------------------
        # Classification
        # -------------------------
        prediction = predict_leaf_class(img)

        pred_class = prediction["predicted_class"]
        pred_class_display = friendly_class_name(pred_class)
        pred_conf = prediction["confidence"]
        pred_idx = prediction["predicted_index"]
        probabilities = prediction["all_probabilities"]
        model = prediction["model"]
        model_input = prediction["model_input"]

        # -------------------------
        # Grad-CAM
        # -------------------------
        heatmap, _ = make_gradcam_heatmap(
            model=model,
            img_array=model_input,
            pred_index=pred_idx
        )
        gradcam_heatmap_rgb, gradcam_overlay_rgb = overlay_gradcam(img, heatmap, alpha=0.4)

        # -------------------------
        # Resize main visuals
        # -------------------------
        original_large = resize_for_display(results["original"], width=430)
        final_large = resize_for_display(results["final_result"], width=430)
        gradcam_overlay_bgr = cv2.cvtColor(gradcam_overlay_rgb, cv2.COLOR_RGB2BGR)
        gradcam_overlay_large = resize_for_display(gradcam_overlay_bgr, width=430)

        # -------------------------
        # Unified Results Summary
        # -------------------------
        st.markdown('<div class="section-title">Results Summary</div>', unsafe_allow_html=True)

        row1 = st.columns(4)
        with row1[0]:
            st.markdown(f"""
            <div class="info-card primary-card">
                <div class="metric-label">Predicted Class</div>
                <div class="metric-value">{pred_class_display}</div>
            </div>
            """, unsafe_allow_html=True)

        with row1[1]:
            st.markdown(f"""
            <div class="info-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{pred_conf * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with row1[2]:
            st.markdown(f"""
            <div class="info-card accent-card">
                <div class="metric-label">Disease Percentage</div>
                <div class="metric-value">{results['disease_percentage']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with row1[3]:
            st.markdown(f"""
            <div class="info-card accent-card">
                <div class="metric-label">Severity</div>
                <div class="metric-value">{results['severity']}</div>
            </div>
            """, unsafe_allow_html=True)

        row2 = st.columns(3)
        with row2[0]:
            st.markdown(f"""
            <div class="info-card quant-card">
                <div class="metric-label">Leaf Area</div>
                <div class="metric-value">{results['leaf_area']} px</div>
            </div>
            """, unsafe_allow_html=True)

        with row2[1]:
            st.markdown(f"""
            <div class="info-card quant-card">
                <div class="metric-label">Diseased Area</div>
                <div class="metric-value">{results['disease_area']} px</div>
            </div>
            """, unsafe_allow_html=True)

        with row2[2]:
            st.markdown(f"""
            <div class="info-card quant-card">
                <div class="metric-label">Healthy Area</div>
                <div class="metric-value">{results['healthy_area']} px</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="small-note">
                Pixel-based analysis estimated <b>{results['leaf_area']}</b> total leaf pixels,
                of which <b>{results['disease_area']}</b> pixels were identified as diseased
                after excluding detected shadow regions.
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------
        # Main Visual Outputs
        # -------------------------
        st.markdown('<div class="section-title">Main Visual Outputs</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(original_large, caption="Original Image", channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(final_large, caption="Detected Disease Output", channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(gradcam_overlay_large, caption="Grad-CAM Overlay", channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------
        # Interpretation
        # -------------------------
        st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="interpret-box">
            The final selected model predicted this leaf as <b>{pred_class_display}</b> with
            <b>{pred_conf * 100:.2f}% confidence</b>. The image-processing pipeline estimated
            that <b>{results['disease_percentage']:.2f}%</b> of the leaf area is diseased,
            corresponding to <b>{results['severity']}</b>. The Grad-CAM visualization highlights
            the image regions that most influenced the model’s classification decision.
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # Secondary outputs
        # -------------------------
        with st.expander("Show Additional Detailed Outputs"):
            st.markdown('<div class="section-title">Additional Visual Outputs</div>', unsafe_allow_html=True)

            leaf_only_large = resize_for_display(results["leaf_only"], width=320)
            leaf_mask_small = resize_for_display(results["leaf_mask"], width=320)
            disease_mask_small = resize_for_display(results["disease_mask"], width=320)
            shadow_mask_small = resize_for_display(results["shadow_mask"], width=320)

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.image(leaf_only_large, caption="Leaf Only", channels="BGR", use_container_width=True)
            with d2:
                st.image(leaf_mask_small, caption="Leaf Mask", clamp=True, use_container_width=True)
            with d3:
                st.image(disease_mask_small, caption="Disease Mask", clamp=True, use_container_width=True)
            with d4:
                st.image(shadow_mask_small, caption="Shadow Mask", clamp=True, use_container_width=True)

            st.markdown('<div class="section-title">Raw Class Probabilities</div>', unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "Class": [friendly_class_name(c) for c in prediction["class_names"]],
                "Probability": [round(float(p), 4) for p in probabilities]
            })
            st.dataframe(prob_df, use_container_width=True)

else:
    st.info("Upload a leaf image to begin analysis.")