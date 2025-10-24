#!/bin/bash

# Brain Tumor Classification Web App Launcher

echo "🧠 Starting Brain Tumor Classification Web Application..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if model exists
if [ ! -f "models/brain_tumor_classifier.h5" ]; then
    echo "⚠️  Warning: Model file not found at models/brain_tumor_classifier.h5"
    echo "   Make sure to train your model first using: python src/train.py"
    echo ""
fi

# Start the application
echo "🚀 Starting Flask application..."
echo "   Access the app at: http://localhost:5000"
echo "   Press Ctrl+C to stop the server"
echo ""
python app.py
