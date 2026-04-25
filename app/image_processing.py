import cv2
import numpy as np


def resize_image(image, max_width=700):
    """
    Resize image while keeping aspect ratio.
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    new_height = int((max_width / w) * h)
    return cv2.resize(image, (max_width, new_height))


def preprocess_image(img):
    """
    Keep preprocessing light so the original color information is preserved.
    """
    img = resize_image(img, max_width=700)
    return img


def extract_leaf_mask(img):
    """
    Use the same leaf detection logic that worked in your old app:
    threshold on the Saturation channel.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(img_hsv)

    _, leaf_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)

    kernel_leaf = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_leaf)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel_leaf)

    return leaf_mask


def detect_shadow_mask(img, leaf_mask):
    """
    Detect very dark regions inside the leaf using the V channel.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(img_hsv)

    shadow_threshold_v = 55
    _, shadow_mask = cv2.threshold(v, shadow_threshold_v, 255, cv2.THRESH_BINARY_INV)

    shadow_mask = cv2.bitwise_and(shadow_mask, leaf_mask)

    kernel_shadow = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel_shadow)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel_shadow)

    return shadow_mask


def extract_disease_mask(img, leaf_mask, shadow_mask=None):
    """
    Detect diseased regions using two HSV ranges:
    1. Yellow/brown disease
    2. Dark necrotic spots

    Then:
    - restrict detection to leaf only
    - remove healthy green regions
    - remove shadow regions
    - clean mask with morphology
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1) Yellow / brown diseased regions
    lower_yellow_brown = np.array([0, 25, 25])
    upper_yellow_brown = np.array([35, 255, 190])

    # 2) Dark necrotic spots
    lower_dark_spot = np.array([0, 20, 20])
    upper_dark_spot = np.array([25, 255, 110])

    mask1 = cv2.inRange(img_hsv, lower_yellow_brown, upper_yellow_brown)
    mask2 = cv2.inRange(img_hsv, lower_dark_spot, upper_dark_spot)

    # Combine both disease masks
    disease_mask = cv2.bitwise_or(mask1, mask2)

    # Keep only disease inside the leaf
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    # Healthy green suppression
    lower_healthy_green = np.array([36, 40, 20])
    upper_healthy_green = np.array([95, 255, 255])

    green_mask = cv2.inRange(img_hsv, lower_healthy_green, upper_healthy_green)
    green_mask = cv2.bitwise_and(green_mask, leaf_mask)

    green_inverse = cv2.bitwise_not(green_mask)
    disease_mask = cv2.bitwise_and(disease_mask, green_inverse)

    # Remove shadow regions if provided
    if shadow_mask is not None:
        shadow_inverse = cv2.bitwise_not(shadow_mask)
        disease_mask = cv2.bitwise_and(disease_mask, shadow_inverse)

    # Clean final disease mask
    kernel_disease = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_disease)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_disease)

    return disease_mask


def calculate_metrics(leaf_mask, disease_mask):
    """
    Calculate leaf area, diseased area, healthy area, and disease percentage.
    """
    leaf_area = cv2.countNonZero(leaf_mask)
    disease_area = cv2.countNonZero(disease_mask)
    healthy_area = max(leaf_area - disease_area, 0)

    if leaf_area == 0:
        disease_percentage = 0.0
    else:
        disease_percentage = (disease_area / leaf_area) * 100.0

    return {
        "leaf_area": int(leaf_area),
        "disease_area": int(disease_area),
        "healthy_area": int(healthy_area),
        "disease_percentage": float(disease_percentage)
    }


def get_severity_label(disease_percentage):
    """
    Convert disease percentage to qualitative severity level.
    """
    if disease_percentage <= 5:
        return "Very Low / Healthy"
    elif disease_percentage <= 20:
        return "Mild"
    elif disease_percentage <= 40:
        return "Moderate"
    else:
        return "Severe"


def create_leaf_only(img, leaf_mask):
    """
    Keep only the detected leaf region.
    """
    leaf_only = cv2.bitwise_and(img, img, mask=leaf_mask)
    return cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB)


def create_overlay(img, disease_mask, disease_percentage, severity, shadow_mask=None):
    """
    Cleaner final visualization:
    Diseased regions = red
    No shadow overlay in final result
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    # Disease region in RED
    overlay[disease_mask == 255] = [255, 0, 0]

    final_result = cv2.addWeighted(img_rgb, 0.8, overlay, 0.2, 0)

    h, w = final_result.shape[:2]

    panel_x1 = 10
    panel_y1 = h - 70
    panel_x2 = 250
    panel_y2 = h - 10

    cv2.rectangle(final_result, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)

    cv2.putText(
        final_result,
        f"Disease: {disease_percentage:.2f}%",
        (panel_x1 + 10, panel_y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        final_result,
        f"Severity: {severity}",
        (panel_x1 + 10, panel_y1 + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return final_result


def process_leaf_image(img):
    """
    Full pipeline:
    1. Preprocess
    2. Detect leaf
    3. Detect shadows inside leaf
    4. Detect disease excluding shadows
    5. Compute metrics
    6. Create visualization
    """
    img_processed = preprocess_image(img)
    img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

    leaf_mask = extract_leaf_mask(img_processed)
    shadow_mask = detect_shadow_mask(img_processed, leaf_mask)
    disease_mask = extract_disease_mask(img_processed, leaf_mask, shadow_mask)

    metrics = calculate_metrics(leaf_mask, disease_mask)
    severity = get_severity_label(metrics["disease_percentage"])

    leaf_only_rgb = create_leaf_only(img_processed, leaf_mask)
    final_result = create_overlay(
        img_processed,
        disease_mask,
        metrics["disease_percentage"],
        severity,
        shadow_mask
    )

    return {
        "original": img_rgb,
        "leaf_mask": leaf_mask,
        "shadow_mask": shadow_mask,
        "leaf_only": leaf_only_rgb,
        "disease_mask": disease_mask,
        "final_result": final_result,
        "leaf_area": metrics["leaf_area"],
        "disease_area": metrics["disease_area"],
        "healthy_area": metrics["healthy_area"],
        "disease_percentage": metrics["disease_percentage"],
        "severity": severity
    }