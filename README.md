<div align="center">

# 🧠 Brain Tumor Classification using Deep Learning

### Explainable AI-powered MRI classification with Grad-CAM, SHAP & segmentation-grounded validation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in--progress-brightgreen.svg)]()

**[📂 Explore the Notebook](./brain_tumor_classification.ipynb) · [📋 Project Roadmap](#-roadmap)**

</div>

---

## 🎯 Project Overview

An ongoing capstone project building a deep learning pipeline that doesn't just classify brain MRI scans — it **explains** its decisions and **proves** those explanations are trustworthy. The system classifies scans into four categories (glioma, meningioma, pituitary tumor, no tumor), visualizes its reasoning via Grad-CAM and SHAP, and — uniquely — quantitatively validates those explanations against real tumor segmentation masks using IoU/Dice overlap scoring.

> 🔍 **What makes this different:** Most XAI-for-medical-imaging projects generate a heatmap and stop there. This one measures whether the heatmap actually overlaps with the real tumor.

## 📊 Current Results

| Metric | Value |
|---|---|
| **Model (baseline)** | VGG16, transfer learning |
| **Test Accuracy** | **82.61%** |
| **Test Loss** | 0.4548 |
| **Classes** | Glioma · Meningioma · Pituitary · No Tumor |
| **Dataset** | [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |

## 🚀 Roadmap

- [x] Baseline CNN + VGG16 transfer learning classifier (82.61% test accuracy)
- [ ] 🔄 EfficientNetB0 rebuild — targeting 90%+ accuracy
- [ ] 🔦 Grad-CAM + SHAP explainability integration
- [ ] 🧩 2D U-Net segmentation on BraTS 2021
- [ ] ✅ Quantitative IoU/Dice validation of heatmaps against ground-truth masks
- [ ] 🌐 Interactive web demo (Streamlit/Gradio)

## 🛠️ Tech Stack

`Python` `TensorFlow / Keras` `NumPy` `Matplotlib` `Kaggle Notebooks`

## 📁 Repository Structure

Brain_tumor_classification/
├── brain_tumor_classification.ipynb # Data pipeline, model, training, evaluation
├── requirements.txt
└── README.md
## ⚡ Getting Started

```bash
git clone https://github.com/Parthiv19M/Brain_tumor_classification.git
cd Brain_tumor_classification
pip install -r requirements.txt
```
Open `brain_tumor_classification.ipynb` and run cells sequentially. Dataset can be attached directly in Kaggle/Colab — no manual download needed.

## 👤 Author

**Parthiv Meduri**
B.Tech CSE, KLH University

</div>
