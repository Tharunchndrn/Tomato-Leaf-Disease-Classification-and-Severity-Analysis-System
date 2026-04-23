import streamlit as st
import cv2
import numpy as np
from image_processing import process_leaf_image

st.set_page_config(page_title="QBR Leaf Analyzer", layout="wide")

st.title("Quantitative Botanical Reasoning (QBR) System")
st.write("Upload a tomato leaf image to analyze diseased regions, remove shadows, and get quantitative output.")


def resize_for_display(image, width=350):
    h, w = image.shape[:2]
    new_height = int((width / w) * h)
    return cv2.resize(image, (width, new_height))


uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read the uploaded image.")
    else:
        results = process_leaf_image(img)

        st.subheader("Quantitative Output")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Leaf Area", f"{results['leaf_area']} px")
        col2.metric("Diseased Area", f"{results['disease_area']} px")
        col3.metric("Healthy Area", f"{results['healthy_area']} px")
        col4.metric("Disease Percentage", f"{results['disease_percentage']:.2f}%")
        col5.metric("Severity", results["severity"])

        st.subheader("Primary Output")

        original_large = resize_for_display(results["original"], width=420)
        final_large = resize_for_display(results["final_result"], width=420)
        leaf_only_large = resize_for_display(results["leaf_only"], width=420)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(original_large, caption="Original Leaf")
        with col_b:
            st.image(final_large, caption="Detected Disease Output")
        with col_c:
            st.image(leaf_only_large, caption="Leaf Only")

        with st.expander("Show Detailed Processing Outputs"):
            leaf_mask_small = resize_for_display(results["leaf_mask"], width=250)
            disease_mask_small = resize_for_display(results["disease_mask"], width=250)
            shadow_mask_small = resize_for_display(results["shadow_mask"], width=250)

            col_d, col_e, col_f = st.columns(3)
            with col_d:
                st.image(leaf_mask_small, caption="Leaf Mask", clamp=True)
            with col_e:
                st.image(disease_mask_small, caption="Disease Mask", clamp=True)
            with col_f:
                st.image(shadow_mask_small, caption="Shadow Mask", clamp=True)

        st.subheader("Interpretation")
        st.write(
            f"The system estimated that **{results['disease_percentage']:.2f}%** of the detected leaf area "
            f"is diseased after excluding detected shadow regions. This is categorized as "
            f"**{results['severity']}**."
        )
else:
    st.info("Upload a leaf image to begin analysis.")