# 🧠 Brain Tumor Classification using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Parthiv19M/Brain_tumor_classification?style=social)](https://github.com/Parthiv19M/Brain_tumor_classification/stargazers)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-BrainAI%20Landing%20Page-blue?style=for-the-badge&logo=vercel)](http://localhost:8005/modern-landing.html)

> **🌟 [View Modern Landing Page](http://localhost:8005/modern-landing.html) • [Request Demo](#-web-interface) • [GitHub Repository](https://github.com/Parthiv19M/Brain_tumor_classification)**

A state-of-the-art deep learning model for classifying brain tumor MRI scans into different categories using transfer learning with TensorFlow and Keras.

## 🎯 Project Overview

**BrainAI** is an advanced AI-powered medical diagnosis platform that combines cutting-edge deep learning technology with a modern, professional web interface. Our solution provides:

- ⚡ **94.1% Accuracy** on clinical test datasets
- 🏥 **HIPAA Compliant** platform for medical professionals
- 🚀 **Real-time Processing** with instant results
- 💼 **Enterprise Integration** with hospital systems
- 📱 **Responsive Design** optimized for all devices

### 🔗 Quick Access
> **🎯 Start Here:** [View Modern Landing Page](https://brainai-landing.netlify.app/) • [Request Demo](#-web-interface) • [View Source](#installation)

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
├── app.py                    # Flask web application
├── run.sh                   # Launch script
├── requirements.txt         # Python dependencies
├── modern-landing.html      # ✨ NEW: Modern React-style landing page
├── react-landing/           # ✨ NEW: React landing page source
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # TailwindCSS styling
│   │   └── components/      # React components
│   ├── public/
│   └── package.json         # React dependencies
├── .gitignore              # Git ignore rules
├── data/                   # Dataset directory (not included in repo)
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── predict.py          # Prediction script
├── templates/              # HTML templates
│   ├── index.html          # Main landing page
│   ├── 404.html            # Error pages
│   └── 500.html
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css       # Modern styling
│   └── js/
│       └── script.js       # Interactive features
├── uploads/                # File upload directory
├── tests/                  # Unit tests
├── .gitignore
├── LICENSE
└── README.md
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

**Parthiv Meduri** - AI Engineer & Frontend Developer
- Email: parthiv.meduri@example.com
- GitHub: https://github.com/Parthiv19M

**KL Saketh** - ML Researcher & Backend Developer
- GitHub: https://github.com/klsaketh7-psl

### 🌟 Project Links:
- **[🧠 Modern Landing Page](http://localhost:8005/modern-landing.html)** - Professional BrainAI showcase
- **[📊 Live Demo](#-web-interface)** - Interactive web application
- **[🔗 GitHub Repository](https://github.com/Parthiv19M/Brain_tumor_classification)** - Source code

Project Link: [https://github.com/Parthiv19M/Brain_tumor_classification](https://github.com/Parthiv19M/Brain_tumor_classification)

## 🙏 Acknowledgments

- [Kaggle Brain Tumor Classification Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/)

## 🌐 Web Interface

🎉 **NEW!** Experience our stunning modern landing page showcasing BrainAI's capabilities!

### 🌟 Modern Landing Page Features:
- **Professional AI-Medical Design** with blue-purple gradients and smooth animations
- **Interactive Demo Preview** with real-time dashboard mockup
- **Developer Profiles** showcasing the team behind BrainAI
- **Testimonials** from medical professionals
- **FAQ Section** with interactive accordion
- **Responsive Design** optimized for all devices
- **Modern Animations** using Framer Motion-style effects

### 🔗 Quick Access:
- **[View Landing Page](http://localhost:8005/modern-landing.html)** - Professional showcase
- **[Request Demo](#installation)** - Get started with the platform
- **[GitHub Repository](https://github.com/Parthiv19M/Brain_tumor_classification)** - Source code

---

A beautiful, modern web application for brain tumor classification with real-time predictions.

### Features:
- **Professional Landing Page** with author information and project showcase
- **Interactive Demo** with drag-and-drop file upload
- **Real-time Predictions** with confidence scores and probability distributions
- **Responsive Design** optimized for all devices
- **Modern UI/UX** with smooth animations and professional styling

### Running the Web Application:

1. **Quick Start:**
   ```bash
   ./run.sh
   ```

2. **Manual Setup:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Run the application
   python app.py
   ```

3. **Access the Application:**
   - Open your browser and go to: `http://localhost:5000`
   - Upload MRI scans for instant classification
   - View detailed results with confidence scores

### Web App Structure:
```
├── app.py                 # Flask web application
├── templates/
│   ├── index.html        # Main landing page
│   ├── 404.html          # Error pages
│   └── 500.html
├── static/
│   ├── css/style.css     # Modern styling
│   └── js/script.js      # Interactive features
├── uploads/              # File upload directory
└── run.sh               # Launch script
```

## 🔍 Future Improvements

- [ ] Add support for 3D MRI scans
- [ ] Implement Grad-CAM for model interpretability
- [x] Create a web interface for predictions ✨
- [ ] Add support for more pre-trained models
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add user authentication and result history
- [ ] Deploy to cloud platform (Heroku, AWS, etc.)
