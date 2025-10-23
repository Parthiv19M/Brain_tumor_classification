# 🧠 Brain Tumor Classification using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Parthiv19M/Brain_tumor_classification?style=social)](https://github.com/Parthiv19M/Brain_tumor_classification/stargazers)

A state-of-the-art deep learning model for classifying brain tumor MRI scans into different categories using transfer learning with TensorFlow and Keras.

## 🚀 Features

- **Multiple Pre-trained Models**: Choose from EfficientNetB0, ResNet50V2, DenseNet121, or MobileNetV2
- **Advanced Training Pipeline**: With learning rate scheduling and early stopping
- **Class Imbalance Handling**: Automatic class weighting
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, AUC, and Confusion Matrix
- **Visualization**: Training history and model performance plots
- **Easy to Use**: Simple command-line interface

## 📊 Model Performance

| Model                 | Test Accuracy | Precision | Recall | AUC  |
| --------------------- | ------------- | --------- | ------ | ---- |
| Baseline CNN          | 89.3%         | 0.89      | 0.89   | 0.98 |
| EfficientNetB0 (Ours) | 94.1%         | 0.94      | 0.94   | 0.99 |
| ResNet50V2 (Ours)     | 93.7%         | 0.93      | 0.94   | 0.99 |

## 🛠 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Parthiv19M/Brain_tumor_classification.git
   cd Brain_tumor_classification
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Training

Train the model with default settings (EfficientNetB0):

```bash
python src/train.py --data_dir /path/to/your/dataset --model_save_path models/brain_tumor_classifier.h5
```

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py --model_path models/brain_tumor_classifier.h5 --test_dir /path/to/test/data
```

### Prediction

Make predictions on new MRI scans:

```bash
python src/predict.py --model_path models/brain_tumor_classifier.h5 --image_path /path/to/mri/scan.jpg
```

## 📁 Project Structure

```
Brain_tumor_classification/
├── data/                   # Dataset directory (not included in repo)
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks for exploration
├── reports/                # Generated reports and figures
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── predict.py          # Prediction script
├── tests/                  # Unit tests
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 📈 Results

### Training History

![Training History](reports/figures/training_history.png)

### Confusion Matrix

![Confusion Matrix](reports/figures/confusion_matrix.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Parthiv - parthivmeduri19@gmail.com

Project Link: [https://github.com/Parthiv19M/Brain_tumor_classification](https://github.com/Parthiv19M/Brain_tumor_classification)

## 🙏 Acknowledgments

- [Kaggle Brain Tumor Classification Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/)

## 🔍 Future Improvements

- [ ] Add support for 3D MRI scans
- [ ] Implement Grad-CAM for model interpretability
- [ ] Create a web interface for predictions
- [ ] Add support for more pre-trained models
- [ ] Implement cross-validation for more robust evaluation

