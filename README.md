<div align="center">

  <h1>🌿 Tomato Leaf Disease Classification and Severity Analysis</h1>

  <p>
    <b>Image Processing • Computer Vision • Deep Learning • Explainable AI</b>
  </p>

  <br/>

  <img src="https://img.shields.io/badge/Project-Image%20Processing%20%7C%20Computer%20Vision%20%7C%20Deep%20Learning-2563eb?style=for-the-badge&logo=opencv&logoColor=white" />
  <br/>
  <img src="https://img.shields.io/badge/Status-Prototype-22c55e?style=for-the-badge&logo=github&logoColor=white" />
  <img src="https://img.shields.io/badge/Best%20Model-EfficientNetB0-7c3aed?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Test%20Accuracy-97.34%25-16a34a?style=for-the-badge&logo=target&logoColor=white" />

  <br/>
  <br/>

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
    <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  </p>

  <p>
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" />
    <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
    <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  </p>

  <br/>

  <table width="90%">
    <tr>
      <td align="center"><b>🌱 Disease Classes</b></td>
      <td align="center"><b>🧠 Models Compared</b></td>
      <td align="center"><b>📊 Best Accuracy</b></td>
      <td align="center"><b>🔍 Interpretability</b></td>
    </tr>
    <tr>
      <td align="center">
        Early Blight<br/>
        Late Blight<br/>
        Healthy
      </td>
      <td align="center">
        Custom CNN<br/>
        MobileNetV2<br/>
        EfficientNetB0
      </td>
      <td align="center">
        <b>97.34%</b><br/>
        EfficientNetB0
      </td>
      <td align="center">
        Grad-CAM<br/>
        Severity Mapping
      </td>
    </tr>
  </table>

</div>

## 📌 Overview

This project is an end-to-end **computer vision system** built for **tomato leaf disease classification** and **severity analysis**.

It goes beyond a basic classifier by combining:
- **Image processing-based severity analysis**
- **Deep learning classification**
- **Explainable AI with Grad-CAM**
- **Prototype application integration**

The system not only predicts the leaf disease class, but also estimates:
- Leaf area
- Diseased area
- Healthy area
- Disease percentage
- Severity level

---

## 📂 Dataset


This project uses the **PlantVillage tomato leaf dataset**.  
For this prototype, the dataset was narrowed down to **3 important classes**:

<div align="center">

| 🌿 Class | 🏷️ Category | 🎯 Purpose |
|---|---|---|
| Early Blight | Diseased | Detect common fungal leaf infection |
| Late Blight | Diseased | Detect severe tomato leaf disease |
| Healthy | Non-diseased | Compare diseased vs normal leaf condition |

</div>

#### Why were only these 3 classes selected?

The project focuses on these classes to keep the system clear, controlled, and suitable for academic experimentation.


## ✨ Key Features
- Maintain a focused and interpretable classification problem
- Clearly compare **diseased vs healthy** tomato leaf conditions
- Support image processing-based **severity analysis**
- Evaluate multiple deep learning models under a controlled setup
- Build a simple but meaningful prototype for disease classification and severity estimation
- Performs **pixel-based severity analysis**
- Measures diseased and healthy regions
- Uses **Grad-CAM** to highlight regions influencing predictions
- Compares multiple deep learning models
- Includes a prototype interface for image upload and analysis

---

## 🧠 Models Compared

This project evaluates three models:
- **Custom CNN**
- **MobileNetV2**
- **EfficientNetB0**

### ✅ Final Selected Model
**EfficientNetB0** (achieving the highest test accuracy of **97.34%**)

---

## 📊 Performance Summary

| Model | Final Train Accuracy | Best Validation Accuracy | Test Accuracy | Test Loss |
|------|----------------------:|-------------------------:|--------------:|----------:|
| Custom CNN | 0.8844 | 0.8919 | 0.8935 | 0.2477 |
| MobileNetV2 | 0.9536 | 0.9452 | 0.9408 | 0.1540 |
| **EfficientNetB0** | **0.9619** | **0.9704** | **0.9734** | **0.0888** |

---

## 🔬 Image Processing Pipeline

The image-processing workflow includes:
1. **Image preprocessing**
2. **Leaf segmentation**
3. **Shadow detection**
4. **Disease region extraction**
5. **Healthy green suppression**
6. **Morphological refinement**
7. **Pixel-based quantitative analysis**
8. **Severity label estimation**
9. **Final disease overlay generation**

### Quantitative outputs produced:
- **Leaf Area (px)**
- **Diseased Area (px)**
- **Healthy Area (px)**
- **Disease Percentage (%)**
- **Severity Label**

---

## 🖼️ Visual Gallery

Below are the visual outputs from the system's analysis pipeline.

<p align="center">
  <img src="outputs/Interface.png" width="45%" alt="Disease Analysis Result" />
  <img src="outputs/Comparison.png" width="45%" alt="Model Comparison Plot" />
</p>

---

## 📁 Project Structure

```text
tomato-leaf-disease-analysis/
│
├── app/
│   ├── app_flask.py           # Main Flask application
│   ├── image_processing.py    # Image processing logic
│   ├── model_inference.py     # Model loading and prediction
│   ├── gradcam.py             # Grad-CAM implementation
│   ├── templates/             # HTML templates
│   └── static/                # Static assets (CSS, JS, results)
│
├── notebooks/
│   ├── model_train.ipynb      # Model training experiments
│   ├── model_evalution.ipynb  # Evaluation metrics and plots
│   └── image_processing.ipynb # Pipeline development
│
├── scripts/
│   ├── datasplit.py           # Dataset partitioning
│   ├── data_augment.py        # Image augmentation
│   └── class_weighting.py     # Handling class imbalance
│
├── outputs/                   # Performance logs and sample images
├── dataset/                   # Training dataset
├── final_model/               # Saved trained models
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Tharunchndrn/quantitative-botanical-reasoning.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app/app_flask.py
   ```

---

## 👨‍💻 Developer
**Tharun Chandran**
