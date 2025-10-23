"""Training script for the enhanced brain tumor classification model with transfer learning."""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from ..data.load_data import load_data, create_data_generators
from .model import create_model, get_class_weights, unfreeze_model

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics.
    
    Args:
        history: Training history from model.fit()
        save_path: Path to save the plot (optional)
    """
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        # Plot training & validation metrics
        axes[i].plot(history.history[metric])
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            axes[i].plot(history.history[val_metric])
            
        axes[i].set_title(f'Model {metric.title()}')
        axes[i].set_ylabel(metric.title())
        axes[i].set_xlabel('Epoch')
        
        # Add legend
        if val_metric in history.history:
            axes[i].legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def train_model(
    data_dir, 
    model_save_path, 
    epochs=30, 
    batch_size=32,
    base_model_name='EfficientNetB0',
    fine_tune_epochs=10,
    learning_rate=1e-4
):
    """Train the enhanced brain tumor classification model.
    
    Args:
        data_dir (str): Path to the data directory
        model_save_path (str): Path to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        base_model_name (str): Name of the base model to use
        fine_tune_epochs (int): Number of epochs for fine-tuning
        learning_rate (float): Initial learning rate
    """
    # Create output directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load and preprocess data
    train_gen, val_gen, test_gen, class_names = create_data_generators(
        data_dir, 
        batch_size=batch_size,
        img_size=(224, 224)  # Most pre-trained models expect 224x224
    )
    
    # Calculate class weights
    class_weights = get_class_weights(train_gen)
    print("\nClass weights:", class_weights)
    
    # Create and compile model
    model = create_model(
        input_shape=(224, 224, 3),
        num_classes=len(class_names),
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight=class_weights
    )
    
    # Fine-tuning
    if fine_tune_epochs > 0:
        print("\nFine-tuning the model...")
        model = unfreeze_model(model, base_layers=100, learning_rate=learning_rate/10)
        
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=history.epoch[-1],
            callbacks=callbacks_list,
            class_weight=class_weights
        )
        
        # Combine histories
        for key in history_fine.history:
            history.history[key].extend(history_fine.history[key])
    
    # Save training history
    with open('reports/training_history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
    
    # Plot training history
    plot_training_history(history, 'reports/figures/training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_gen, return_dict=True)
    
    # Generate predictions
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, 'reports/figures/confusion_matrix.png')
    
    # Save test results
    with open('reports/test_results.json', 'w') as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f)
    
    return history, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train brain tumor classification model')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to the data directory')
    parser.add_argument('--model_save_path', type=str, default='models/brain_tumor_model.h5',
                       help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--base_model', type=str, default='EfficientNetB0',
                       choices=['EfficientNetB0', 'ResNet50V2', 'DenseNet121', 'MobileNetV2'],
                       help='Base model architecture')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                       help='Number of epochs for fine-tuning')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_model_name=args.base_model,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate
    )
