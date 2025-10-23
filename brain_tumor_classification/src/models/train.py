"""Training script for the brain tumor classification model."""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import callbacks
from ..data.load_data import load_data
from .model import create_model

def train_model(data_dir, model_save_path, epochs=20, batch_size=32):
    """Train the brain tumor classification model.
    
    Args:
        data_dir (str): Path to the data directory
        model_save_path (str): Path to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    # Load and preprocess data
    train_ds, val_ds, test_ds = load_data(data_dir, batch_size=batch_size)
    
    # Create and compile model
    model = create_model()
    
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
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    return history, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train brain tumor classification model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the data directory')
    parser.add_argument('--model_save_path', type=str, default='models/brain_tumor_model.h5',
                        help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create directory for saving the model if it doesn't exist
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Train the model
    train_model(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
