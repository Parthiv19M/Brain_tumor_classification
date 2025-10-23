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

| Model | Test Accuracy | Precision | Recall | AUC |
|-------|--------------|-----------|--------|-----|
| Baseline CNN | 89.3% | 0.89 | 0.89 | 0.98 |
| EfficientNetB0 (Ours) | 94.1% | 0.94 | 0.94 | 0.99 |
| ResNet50V2 (Ours) | 93.7% | 0.93 | 0.94 | 0.99 |

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Parthiv19M/Brain_tumor_classification.git
   cd Brain_tumor_classification
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Training

Train the model with default settings (EfficientNetB0):
```bash
python -m src.models.train \
  --data_dir /path/to/your/data \
  --model_save_path models/brain_tumor_model.h5
```

### Advanced Training Options

```bash
python -m src.models.train \
  --data_dir /path/to/your/data \
  --model_save_path models/brain_tumor_model.h5 \
  --epochs 30 \
  --batch_size 32 \
  --base_model ResNet50V2 \
  --fine_tune_epochs 10 \
  --learning_rate 1e-4
```

### Available Models
- `EfficientNetB0` (default)
- `ResNet50V2`
- `DenseNet121`
- `MobileNetV2`

## 📂 Project Structure

```
Brain_tumor_classification/
├── data/                    # Dataset directory (not included in repo)
├── models/                  # Saved models
├── notebooks/               # Jupyter notebooks for exploration
├── reports/                 # Generated reports and figures
│   └── figures/             # Training plots and confusion matrices
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   └── models/              # Model definitions and training
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt         # Python dependencies
└── setup.py                # Package configuration
```

## 📈 Training Visualization

### Training History
![Training History](reports/figures/training_history.png)

### Confusion Matrix
![Confusion Matrix](reports/figures/confusion_matrix.png)

## 🧪 Testing the Model

After training, you can evaluate the model on the test set:

```python
import tensorflow as tf
from src.models.model import create_model

# Load the trained model
model = tf.keras.models.load_model('models/brain_tumor_model.h5')

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Brain Tumor Classification (MRI)](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- TensorFlow Team for the amazing deep learning framework
- All contributors who helped improve this project

---

<div align="center">
  Made with ❤️ by Parthiv | <a href="https://github.com/Parthiv19M">GitHub</a>
</div>
