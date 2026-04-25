import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect

from image_processing import process_leaf_image
from model_inference import predict_leaf_class
from gradcam import make_gradcam_heatmap, overlay_gradcam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def save_image(path, image):
    cv2.imwrite(path, image)


def friendly_class_name(class_name):
    return class_name.replace("Tomato_", "").replace("_", " ").title()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "leaf_image" not in request.files:
            return redirect(request.url)

        file = request.files["leaf_image"]

        if file.filename == "":
            return redirect(request.url)

        unique_id = str(uuid.uuid4())[:8]
        upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_original.jpg")

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return render_template("index.html", error="Could not read the uploaded image.")

        save_image(upload_path, img)

        # -------------------------
        # Image processing
        # -------------------------
        results = process_leaf_image(img)

        detected_path = os.path.join(RESULT_FOLDER, f"{unique_id}_detected.jpg")
        leaf_only_path = os.path.join(RESULT_FOLDER, f"{unique_id}_leaf_only.jpg")
        leaf_mask_path = os.path.join(RESULT_FOLDER, f"{unique_id}_leaf_mask.jpg")
        disease_mask_path = os.path.join(RESULT_FOLDER, f"{unique_id}_disease_mask.jpg")
        shadow_mask_path = os.path.join(RESULT_FOLDER, f"{unique_id}_shadow_mask.jpg")

        save_image(detected_path, results["final_result"])
        save_image(leaf_only_path, results["leaf_only"])
        save_image(leaf_mask_path, results["leaf_mask"])
        save_image(disease_mask_path, results["disease_mask"])
        save_image(shadow_mask_path, results["shadow_mask"])

        # -------------------------
        # Model prediction
        # -------------------------
        prediction = predict_leaf_class(img)
        pred_class = prediction["predicted_class"]
        pred_idx = prediction["predicted_index"]
        pred_conf = prediction["confidence"]
        model = prediction["model"]
        model_input = prediction["model_input"]
        probabilities = prediction["all_probabilities"]
        class_names = prediction["class_names"]

        # -------------------------
        # Grad-CAM
        # -------------------------
        heatmap, _ = make_gradcam_heatmap(
            model=model,
            img_array=model_input,
            pred_index=pred_idx
        )
        _, gradcam_overlay_rgb = overlay_gradcam(img, heatmap, alpha=0.4)

        gradcam_overlay_bgr = cv2.cvtColor(gradcam_overlay_rgb, cv2.COLOR_RGB2BGR)
        gradcam_overlay_path = os.path.join(RESULT_FOLDER, f"{unique_id}_gradcam_overlay.jpg")
        save_image(gradcam_overlay_path, gradcam_overlay_bgr)

        prob_data = []
        for cls, prob in zip(class_names, probabilities):
            prob_data.append({
                "class_name": friendly_class_name(cls),
                "probability": round(float(prob) * 100, 2)
            })

        output_data = {
            "predicted_class": friendly_class_name(pred_class),
            "confidence": round(pred_conf * 100, 2),
            "disease_percentage": round(results["disease_percentage"], 2),
            "severity": results["severity"],
            "leaf_area": results["leaf_area"],
            "disease_area": results["disease_area"],
            "healthy_area": results["healthy_area"],
            "original_image": upload_path.replace("\\", "/"),
            "detected_image": detected_path.replace("\\", "/"),
            "gradcam_overlay": gradcam_overlay_path.replace("\\", "/"),
            "leaf_only_image": leaf_only_path.replace("\\", "/"),
            "leaf_mask_image": leaf_mask_path.replace("\\", "/"),
            "disease_mask_image": disease_mask_path.replace("\\", "/"),
            "shadow_mask_image": shadow_mask_path.replace("\\", "/"),
            "probabilities": prob_data
        }

        return render_template("index.html", results=output_data)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)