"""Model architecture for brain tumor classification with transfer learning."""

import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers

def build_model(input_shape=(150, 150, 3), num_classes=4, base_model_name='EfficientNetB0'):
    """Build an enhanced CNN model with transfer learning for brain tumor classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        base_model_name (str): Name of the pre-trained model to use
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    # Select base model
    base_models = {
        'EfficientNetB0': (applications.EfficientNetB0, 512),
        'ResNet50V2': (applications.ResNet50V2, 2048),
        'DenseNet121': (applications.DenseNet121, 1024),
        'MobileNetV2': (applications.MobileNetV2, 1280)
    }
    
    base_model_class, top_dims = base_models.get(base_model_name, base_models['EfficientNetB0'])
    
    # Initialize base model with pre-trained weights
    base_model = base_model_class(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model with enhanced architecture
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomBrightness(0.2)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classifier head with regularization
    x = layers.Dense(
        top_dims // 2, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(
        top_dims // 4, 
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def create_model(input_shape=(150, 150, 3), num_classes=4, learning_rate=1e-4, class_weights=None):
    """Create and compile an enhanced CNN model for brain tumor classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        learning_rate (float): Learning rate for the optimizer
        class_weights (dict): Optional dictionary mapping class indices to weights
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = build_model(input_shape, num_classes)
    
    # Optimizer with weight decay
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=1e-4
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
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
    base_model = None
    for layer in model.layers:
        if 'resnet' in layer.name or 'efficientnet' in layer.name or 'densenet' in layer.name:
            base_model = layer
            break
    
    if base_model is not None:
        # Unfreeze top layers
        for layer in base_model.layers[-base_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=1e-5
        ),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def get_class_weights(generator):
    """Calculate class weights to handle class imbalance.
    
    Args:
        generator: Keras ImageDataGenerator
        
    Returns:
        dict: Class weights for training
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get class labels
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    
    return dict(enumerate(class_weights))
