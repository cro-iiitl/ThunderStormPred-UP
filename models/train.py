import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from config import PROCESSED_DATA_DIR, MODEL_DIR, SEQUENCE_LENGTH, ENHANCED_FEATURES

def load_data(max_files=5):
    """Load preprocessed data for training"""
    # Find the latest sequence files
    import glob
    sequence_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, 'time_series_sequences_*.npz')), reverse=True)
    
    if not sequence_files:
        raise FileNotFoundError(f"No sequence data files found in {PROCESSED_DATA_DIR}")
    
    # Load the data from the files (limited by max_files)
    X_list = []
    y_list = []
    
    for file_path in sequence_files[:max_files]:
        try:
            print(f"Loading sequences from {os.path.basename(file_path)}...")
            data = np.load(file_path)
            X = data['X']
            y = data['y']
            
            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not X_list:
        raise ValueError("No valid data loaded from any files")
    
    # Combine all data
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"Loaded {X.shape[0]} total sequences")
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.2):
    """Split data into training, validation, and test sets"""
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split into train and temp (val + test)
    train_size = int(X.shape[0] * (1 - test_size - val_size))
    X_train, X_temp = X[:train_size], X[train_size:]
    y_train, y_temp = y[:train_size], y[train_size:]
    
    # Split temp into val and test
    val_samples = int(X.shape[0] * val_size)
    X_val, X_test = X_temp[:val_samples], X_temp[val_samples:]
    y_val, y_test = y_temp[:val_samples], y_temp[val_samples:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_enhanced_model(sequence_length, n_features):
    """Create an enhanced deep learning model for thunderstorm prediction"""
    model = Sequential([
        # 1D Convolution layer to capture local patterns
        Conv1D(filters=64, kernel_size=3, activation='relu', 
               input_shape=(sequence_length, n_features)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second Conv1D layer
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.2),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary output for lightning probability
    ])
    
    # Use focal loss for imbalanced classes (thunderstorms are rare events)
    def focal_loss(gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + 1e-7)) - \
                   tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + 1e-7))
        return focal_loss_fixed
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(),
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(continue_training=False):
    """Train the thunderstorm prediction model"""
    print("Loading data...")
    X, y = load_data()
    
    print("Splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    # Count class distribution
    positive_samples = np.sum(y_train == 1)
    total_samples = len(y_train)
    print(f"Class distribution - Positive: {positive_samples} ({positive_samples/total_samples:.2%}), Negative: {total_samples - positive_samples} ({1 - positive_samples/total_samples:.2%})")
    
    if continue_training and os.path.exists(os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5')):
        print("Loading existing model...")
        model = load_model(os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5'), compile=False)
        
        # Recompile with focal loss
        def focal_loss(gamma=2., alpha=.25):
            def focal_loss_fixed(y_true, y_pred):
                pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
                pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
                return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + 1e-7)) - \
                       tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + 1e-7))
            return focal_loss_fixed
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for fine-tuning
            loss=focal_loss(),
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    else:
        print("Building new model...")
        model = build_enhanced_model(SEQUENCE_LENGTH, len(ENHANCED_FEATURES))
    
    model.summary()
    
    # Setup callbacks
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.h5')
    callbacks = [
        ModelCheckpoint(
            checkpoint_path, 
            monitor='val_auc', 
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )
    ]
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight={0: 1, 1: 3}  # Balance classes by giving more weight to positive samples
    )
    
    # Evaluate on test set
    print("Evaluating model...")
    test_results = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")
    print(f"Test precision: {test_results[3]:.4f}")
    print(f"Test recall: {test_results[4]:.4f}")
    
    # Save the final model
    model.save(os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5'))
    print(f"Model saved to {os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5')}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Training and Validation AUC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.close()
    
    return model, history

def feature_importance():
    """Key features for thunderstorm prediction and their importance"""
    # Based on meteorological knowledge
    return {
        'cape': 9,           # Convective Available Potential Energy - crucial for thunderstorm formation
        'temperature': 7,    # Temperature gradient
        'humidity': 8,       # High humidity needed for condensation
        'pressure': 5,       # Pressure systems
        'wind_speed': 6,     # Wind affects storm development
        'wind_direction': 4, # Wind direction
        'precipitable_water': 7, # Total column moisture
        'lifted_index': 8,   # Stability index
        'k_index': 7,        # Composite index for thunderstorm potential
        'dewpoint': 8,       # Dewpoint depression - crucial for thunderstorm formation
        'cloud_cover': 6,    # Cloud formation
        'precipitation': 5,  # Precipitation history
    }

if __name__ == "__main__":
    train_model()