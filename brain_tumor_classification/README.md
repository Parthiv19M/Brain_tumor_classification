# 🧠 Brain Tumor Classification using Deep Learning

![Training Progress](https://github.com/Parthiv19M/Brain_tumor_classification/assets/your_username/your_image.png)

This project implements a deep learning model for classifying different types of brain tumors from MRI scans using TensorFlow/Keras. The model can classify brain tumors into four categories: Glioma, Meningioma, No tumor, and Pituitary.

## 📊 Results

### Training Progress
- **Epochs**: 10
- **Final Training Accuracy**: 99.4%
- **Final Validation Accuracy**: 96.9%
- **Final Test Accuracy**: 95.6%

### Confusion Matrix
```
              precision    recall  f1-score   support

      Glioma       0.97      0.96      0.96       300
 Meningioma       0.96      0.95      0.95       306
   No tumor       0.95      0.98      0.96       405
  Pituitary       0.95      0.94      0.94       300

    accuracy                           0.96      1311
   macro avg       0.96      0.96      0.96      1311
weighted avg       0.96      0.96      0.96      1311
```

## 🧠 Project Overview
- **Model**: Deep Learning (TensorFlow/Keras)
- **Architecture**: Custom CNN with Data Augmentation
- **Input**: MRI scans (150x150 pixels, RGB)
- **Output**: Classification into one of four categories
- **Classes**: Glioma, Meningioma, No tumor, Pituitary

## 🚀 Features
- Data augmentation for better model generalization
- Transfer learning with fine-tuning
- Detailed Jupyter notebook with step-by-step implementation
- Model evaluation and visualization tools
- Easy-to-use prediction interface

## 🛠️ Installation

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

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

### Using the Jupyter Notebook
```bash
jupyter notebook notebooks/brain_tumor_classification.ipynb
```

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
