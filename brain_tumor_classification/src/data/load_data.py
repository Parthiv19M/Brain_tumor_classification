"""Data loading and preprocessing for brain tumor classification."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(data_dir, batch_size=32, img_size=(150, 150), validation_split=0.2):
    """Create data generators for training, validation, and testing.
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for the data generators
        img_size (tuple): Target image size (height, width)
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (train_generator, validation_generator, test_generator, class_names)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation and test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'Training'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation data generator
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'Training'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test data generator
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'Testing'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, test_generator, class_names

def load_data(data_dir, batch_size=32, img_size=(150, 150), validation_split=0.2):
    """Load and preprocess the dataset.
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for the data loaders
        img_size (tuple): Target image size (height, width)
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    train_gen, val_gen, test_gen, class_names = create_data_generators(
        data_dir, batch_size, img_size, validation_split
    )
    
    return train_gen, val_gen, test_gen, class_names

def get_class_weights(generator):
    """Calculate class weights to handle class imbalance.
    
    Args:
        generator: Keras data generator
        
    Returns:
        dict: Class weights for training
    """
    class_counts = np.bincount(generator.classes)
    total_samples = len(generator.classes)
    num_classes = len(np.unique(generator.classes))
    
    class_weights = {
        i: total_samples / (num_classes * count) 
        for i, count in enumerate(class_counts)
    }
    
    return class_weights
