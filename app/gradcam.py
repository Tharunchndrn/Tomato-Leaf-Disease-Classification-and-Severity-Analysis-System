import cv2
import numpy as np
import tensorflow as tf


def get_last_conv_layer_name(model):
    """
    Automatically find the last Conv2D layer in the model.
    This avoids hardcoding the EfficientNet last conv layer name.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None, pred_index=None):
    """
    Generates Grad-CAM heatmap for a given preprocessed image array.

    Args:
        model: loaded keras model
        img_array: shape (1, 224, 224, 3)
        last_conv_layer_name: optional; if None, auto-detect
        pred_index: optional class index; if None, use top predicted class

    Returns:
        heatmap: normalized heatmap in range [0,1]
        pred_index: predicted class index used for Grad-CAM
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap /= max_val

    return heatmap, int(pred_index)


def overlay_gradcam(original_bgr, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap onto the original BGR image.

    Args:
        original_bgr: original image in BGR format
        heatmap: Grad-CAM heatmap in [0,1]
        alpha: overlay intensity

    Returns:
        heatmap_colored_rgb: colored heatmap for display
        overlay_rgb: overlay image for display
    """
    h, w = original_bgr.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return heatmap_colored_rgb, overlay_rgb