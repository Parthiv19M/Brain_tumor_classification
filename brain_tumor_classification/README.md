# 🧠 Brain Tumor Classification using Deep Learning

![Training Progress](https://github.com/Parthiv19M/Brain_tumor_classification/assets/your_username/your_image.png)

This project implements a deep learning model for classifying different types of brain tumors from MRI scans using TensorFlow/Keras. The model can classify brain tumors into four categories: Glioma, Meningioma, No tumor, and Pituitary.

## 📊 Results

### Training Progress
- **Epochs**: 10
- **Final Training Accuracy**: 99.4%
- **Final Validation Accuracy**: 96.9%
- **Final Test Accuracy**: 95.6%


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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Training

### Training the Model
```bash
python -m src.models.train --data_dir /path/to/your/data --epochs 20 --batch_size 32
```

### Making Predictions
```python
from src.models.model import create_model
from src.data.load_data import load_image

# Load the trained model
model = tf.keras.models.load_model('models/brain_tumor_model.h5')

# Load and preprocess image
image = load_image('path_to_image.jpg')
prediction = model.predict(image)
```

## 📊 Dataset

The dataset contains MRI scans of brain tumors categorized into four classes:
- Glioma (827 images)
- Meningioma (822 images)
- No tumor (395 images)
- Pituitary (827 images)

### Data Distribution
- **Training Set**: 2,871 images
- **Validation Set**: 717 images
- **Test Set**: 1,311 images

## 🏗️ Project Structure
```
brain_tumor_classification/
├── data/                 # Dataset directory
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   └── models/          # Model architecture and training
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Dataset: [Brain Tumor Classification (MRI)](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- TensorFlow and Keras documentation
- The open-source community for their valuable contributions
