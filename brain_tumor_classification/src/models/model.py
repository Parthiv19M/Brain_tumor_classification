"""Model architecture for brain tumor classification."""

import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_model(input_shape=(150, 150, 3), num_classes=4):
    """Build a CNN model for brain tumor classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    # Base model (you can replace this with a pre-trained model like ResNet50)
    base_model = applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        # Data augmentation layers
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.2),
        
        # Base model
        base_model,
        
        # Classifier
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_model(input_shape=(150, 150, 3), num_classes=4, learning_rate=1e-4):
    """Create and compile a CNN model for brain tumor classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        learning_rate (float): Learning rate for the optimizer
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = build_model(input_shape, num_classes)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_model(model, base_layers=100, learning_rate=1e-5):
    """Unfreeze some layers of the base model for fine-tuning.
    
    Args:
        model: Compiled Keras model
        base_layers: Number of layers to unfreeze from the top of the base model
        learning_rate: New learning rate for fine-tuning
        
    Returns:
        tf.keras.Model: Model with unfrozen layers
    """
    # Unfreeze the base model
    for layer in model.layers[3].layers[-base_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
