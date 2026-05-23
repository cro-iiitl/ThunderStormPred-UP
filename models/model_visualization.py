# models/model_visualization.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import random

# Create project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model parameters
SEQUENCE_LENGTH = 24
FEATURES = [
    'temperature', 'humidity', 'pressure', 'wind_speed', 
    'wind_direction', 'precipitation', 'cloud_cover'
]

# Set the number of samples to generate (1 lakh = 100,000)
NUM_SAMPLES = 100000

def generate_thesis_data(samples=NUM_SAMPLES):
    """Generate realistic data for thesis demonstration"""
    print(f"Generating realistic training data for thesis demonstration ({samples} samples)...")
    
    n_features = len(FEATURES)
    X = np.zeros((samples, SEQUENCE_LENGTH, n_features))
    y = np.zeros(samples)
    
    # Feature indices based on typical order
    temp_idx = 0      # temperature
    hum_idx = 1       # humidity
    press_idx = 2     # pressure
    wind_s_idx = 3    # wind speed
    wind_d_idx = 4    # wind direction
    precip_idx = 5    # precipitation
    cloud_idx = 6     # cloud cover
    
    # Batch processing to avoid memory issues
    batch_size = 10000
    num_batches = (samples + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, samples)
        batch_samples = end_idx - start_idx
        
        print(f"Generating batch {batch+1}/{num_batches} ({batch_samples} samples)...")
        
        # Generate sequences with meteorological patterns
        for i in range(start_idx, end_idx):
            local_i = i - start_idx  # Local index within the batch
            
            # Base pattern with random variations
            base_temp = np.random.uniform(20, 35)  # Base temperature
            
            # Generate sequence (24 hours)
            for t in range(SEQUENCE_LENGTH):
                # Daily temperature cycle
                hour_of_day = t % 24
                temp_cycle = 5 * np.sin(np.pi * hour_of_day / 12)  # +/- 5°C daily cycle
                
                # Temperature with some random variations
                X[i, t, temp_idx] = base_temp + temp_cycle + np.random.normal(0, 0.5)
                
                # Humidity (inversely related to temperature)
                X[i, t, hum_idx] = 100 - (X[i, t, temp_idx] - 20) * 2 + np.random.normal(0, 5)
                X[i, t, hum_idx] = np.clip(X[i, t, hum_idx], 30, 100)
                
                # Air pressure (dropping pressure often precedes storms)
                if i % 4 == 0:  # 25% chance of pressure drop pattern
                    X[i, t, press_idx] = 1013 - t * 0.5 + np.random.normal(0, 0.3)  # Dropping pressure
                else:
                    X[i, t, press_idx] = 1013 + np.random.normal(0, 1)  # Stable pressure
                
                # Wind speed (increases before storms)
                if i % 4 == 0:  # Aligned with pressure drop
                    X[i, t, wind_s_idx] = 5 + t * 0.3 + np.random.normal(0, 0.5)  # Increasing wind
                else:
                    X[i, t, wind_s_idx] = np.random.uniform(0, 10)  # Normal wind
                
                # Wind direction
                X[i, t, wind_d_idx] = np.random.uniform(0, 360)
                
                # Precipitation (more likely with higher humidity and dropping pressure)
                if X[i, t, hum_idx] > 80 and X[i, t, press_idx] < 1010:
                    X[i, t, precip_idx] = np.random.exponential(3)  # Heavier precipitation
                else:
                    X[i, t, precip_idx] = np.random.exponential(0.5)  # Light or no precipitation
                
                # Cloud cover (correlates with humidity)
                X[i, t, cloud_idx] = X[i, t, hum_idx] * 0.8 + np.random.normal(0, 10)
                X[i, t, cloud_idx] = np.clip(X[i, t, cloud_idx], 0, 100)
            
            # Determine storm likelihood based on patterns
            # High humidity + dropping pressure + increasing wind = higher storm chance
            avg_humidity = np.mean(X[i, :, hum_idx])
            pressure_drop = X[i, 0, press_idx] - X[i, -1, press_idx]
            wind_increase = X[i, -1, wind_s_idx] - X[i, 0, wind_s_idx]
            
            # Calculate storm probability
            storm_prob = 0.1  # Base probability
            
            if avg_humidity > 85:  # High humidity
                storm_prob += 0.3
            elif avg_humidity > 75:
                storm_prob += 0.15
                
            if pressure_drop > 5:  # Significant pressure drop
                storm_prob += 0.3
            elif pressure_drop > 2:
                storm_prob += 0.15
                
            if wind_increase > 5:  # Significant wind increase
                storm_prob += 0.2
            elif wind_increase > 2:
                storm_prob += 0.1
                
            # Set label based on probability (binary classification)
            y[i] = 1 if np.random.random() < storm_prob else 0
    
    # Make sure we have some positive examples (at least 20%)
    positive_count = np.sum(y)
    positive_rate = positive_count / samples
    print(f"Initial storm rate: {positive_rate:.2%}")
    
    if positive_rate < 0.2:
        # Add some additional positive examples to balance the dataset
        num_to_add = int(samples * 0.2 - positive_count)
        print(f"Adding {num_to_add} storm examples to reach 20%...")
        
        # Find indices of non-storm examples to convert
        non_storm_indices = np.where(y == 0)[0]
        indices_to_convert = np.random.choice(non_storm_indices, size=num_to_add, replace=False)
        
        for idx in indices_to_convert:
            # Create a storm pattern
            for t in range(SEQUENCE_LENGTH):
                X[idx, t, hum_idx] = 80 + t * 0.5  # Increasing humidity
                X[idx, t, press_idx] = 1010 - t * 0.6  # Dropping pressure
                X[idx, t, wind_s_idx] = 8 + t * 0.4  # Increasing wind
                X[idx, t, cloud_idx] = 70 + t * 1.0  # Increasing cloud cover
                if t > 18:  # Precipitation in last hours
                    X[idx, t, precip_idx] = 5 + (t-18) * 2
            y[idx] = 1  # Mark as storm
    
    # Print class distribution info
    storm_count = np.sum(y)
    print(f"Final dataset: {samples} samples with {storm_count} storms ({storm_count/samples:.1%} storm rate)")
    
    # Save the generated data
    sequence_file = os.path.join(PROCESSED_DATA_DIR, 'thesis_training_sequences.npz')
    print(f"Saving generated data to {sequence_file}...")
    np.savez(sequence_file, X=X, y=y)
    print(f"Data saved successfully!")
    
    # Also save a sample as CSV for visualization
    sample_df = pd.DataFrame()
    for i, feature in enumerate(FEATURES):
        sample_df[feature] = X[0, :, i]
    sample_df['hour'] = range(24)
    sample_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'sample_sequence.csv'), index=False)
    
    return X, y

def visualize_sample_sequence():
    """Visualize a sample sequence to explain the input data"""
    # Load sample sequence
    try:
        sample_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'sample_sequence.csv'))
    except:
        # Generate data if it doesn't exist
        generate_thesis_data(samples=10)
        sample_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'sample_sequence.csv'))
    
    # Create multi-panel plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Temperature and humidity
    ax1 = axes[0]
    color = 'tab:red'
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(sample_df['hour'], sample_df['temperature'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax1_twin = ax1.twinx()
    color = 'tab:blue'
    ax1_twin.set_ylabel('Humidity (%)', color=color)
    ax1_twin.plot(sample_df['hour'], sample_df['humidity'], color=color, marker='s')
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Temperature and Humidity over 24 Hours', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Pressure
    ax2 = axes[1]
    ax2.plot(sample_df['hour'], sample_df['pressure'], color='purple', marker='o')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title('Atmospheric Pressure over 24 Hours', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Wind
    ax3 = axes[2]
    color = 'tab:green'
    ax3.set_ylabel('Wind Speed (m/s)', color=color)
    ax3.plot(sample_df['hour'], sample_df['wind_speed'], color=color, marker='o')
    ax3.tick_params(axis='y', labelcolor=color)
    
    ax3_twin = ax3.twinx()
    color = 'tab:gray'
    ax3_twin.set_ylabel('Wind Direction (°)', color=color)
    ax3_twin.scatter(sample_df['hour'], sample_df['wind_direction'], color=color, alpha=0.5)
    ax3_twin.tick_params(axis='y', labelcolor=color)
    ax3.set_title('Wind Speed and Direction over 24 Hours', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Precipitation and cloud cover
    ax4 = axes[3]
    color = 'tab:blue'
    ax4.set_ylabel('Precipitation (mm)', color=color)
    ax4.plot(sample_df['hour'], sample_df['precipitation'], color=color, marker='o')
    ax4.tick_params(axis='y', labelcolor=color)
    
    ax4_twin = ax4.twinx()
    color = 'tab:gray'
    ax4_twin.set_ylabel('Cloud Cover (%)', color=color)
    ax4_twin.plot(sample_df['hour'], sample_df['cloud_cover'], color=color, marker='s')
    ax4_twin.tick_params(axis='y', labelcolor=color)
    ax4.set_title('Precipitation and Cloud Cover over 24 Hours', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # X axis label
    ax4.set_xlabel('Hour of Day', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'sample_sequence_visualization.png'))
    plt.close()
    
    print(f"Saved sample sequence visualization to {os.path.join(MODEL_DIR, 'sample_sequence_visualization.png')}")

def create_storm_vs_no_storm_comparison():
    """Create a visualization comparing storm vs. no storm patterns"""
    # Generate two specific sample sequences
    X_samples = np.zeros((2, SEQUENCE_LENGTH, len(FEATURES)))
    
    # Sample 1: Storm pattern (high humidity, falling pressure, rising wind)
    for t in range(SEQUENCE_LENGTH):
        # Temperature with daily cycle
        X_samples[0, t, 0] = 28 - 5 * np.sin(np.pi * t / 12)  # Cooler in afternoon (unusual)
        # Humidity increasing
        X_samples[0, t, 1] = 70 + t * 0.8
        # Pressure dropping
        X_samples[0, t, 2] = 1010 - t * 0.4
        # Wind speed increasing
        X_samples[0, t, 3] = 5 + t * 0.3
        # Wind direction shifting
        X_samples[0, t, 4] = (180 + t * 5) % 360
        # Precipitation increasing
        X_samples[0, t, 5] = max(0, t * 0.2 - 3) if t > 15 else 0
        # Cloud cover increasing
        X_samples[0, t, 6] = min(100, 50 + t * 2)
    
    # Sample 2: No storm pattern (moderate humidity, stable pressure, low wind)
    for t in range(SEQUENCE_LENGTH):
        # Normal temperature cycle
        X_samples[1, t, 0] = 30 + 5 * np.sin(np.pi * t / 12)  # Warmer in afternoon
        # Moderate humidity
        X_samples[1, t, 1] = 50 + 10 * np.sin(np.pi * t / 12)  # Slightly lower in afternoon
        # Stable pressure
        X_samples[1, t, 2] = 1015 + np.random.normal(0, 0.2)
        # Low wind speed
        X_samples[1, t, 3] = 3 + np.random.normal(0, 0.5)
        # Consistent wind direction
        X_samples[1, t, 4] = 90 + np.random.normal(0, 10)
        # No precipitation
        X_samples[1, t, 5] = 0
        # Low to moderate cloud cover
        X_samples[1, t, 6] = 30 + 10 * np.sin(np.pi * t / 12)  # Varying slightly
    
    # Create side-by-side plots
    fig, axes = plt.subplots(4, 2, figsize=(20, 16), sharex=True)
    plt.suptitle('Comparison: Storm Pattern vs. No Storm Pattern', fontsize=16)
    
    sample_labels = ['Storm Pattern', 'No Storm Pattern']
    
    for col in range(2):
        # Add title
        axes[0, col].set_title(f"{sample_labels[col]}", fontsize=14)
        
        # Temperature and humidity
        ax1 = axes[0, col]
        color = 'tab:red'
        ax1.set_ylabel('Temperature (°C)', color=color)
        ax1.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 0], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1_twin = ax1.twinx()
        color = 'tab:blue'
        ax1_twin.set_ylabel('Humidity (%)', color=color)
        ax1_twin.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 1], color=color, marker='s')
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Pressure
        ax2 = axes[1, col]
        ax2.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 2], color='purple', marker='o')
        ax2.set_ylabel('Pressure (hPa)')
        ax2.grid(True, alpha=0.3)
        
        # Wind
        ax3 = axes[2, col]
        color = 'tab:green'
        ax3.set_ylabel('Wind Speed (m/s)', color=color)
        ax3.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 3], color=color, marker='o')
        ax3.tick_params(axis='y', labelcolor=color)
        
        ax3_twin = ax3.twinx()
        color = 'tab:gray'
        ax3_twin.set_ylabel('Wind Direction (°)', color=color)
        ax3_twin.scatter(range(SEQUENCE_LENGTH), X_samples[col, :, 4], color=color, alpha=0.5)
        ax3_twin.tick_params(axis='y', labelcolor=color)
        ax3.grid(True, alpha=0.3)
        
        # Precipitation and cloud cover
        ax4 = axes[3, col]
        color = 'tab:blue'
        ax4.set_ylabel('Precipitation (mm)', color=color)
        ax4.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 5], color=color, marker='o')
        ax4.tick_params(axis='y', labelcolor=color)
        
        ax4_twin = ax4.twinx()
        color = 'tab:gray'
        ax4_twin.set_ylabel('Cloud Cover (%)', color=color)
        ax4_twin.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 6], color=color, marker='s')
        ax4_twin.tick_params(axis='y', labelcolor=color)
        ax4.grid(True, alpha=0.3)
        
        # X axis label
        ax4.set_xlabel('Hour of Sequence', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    comparison_path = os.path.join(MODEL_DIR, 'storm_pattern_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    print(f"Saved storm pattern comparison to {comparison_path}")

def build_model(sequence_length, n_features):
    """Create a deep learning model for thunderstorm prediction with visualization"""
    # Define the model
    model = Sequential([
        # First LSTM layer
        Bidirectional(LSTM(128, return_sequences=True), 
                     input_shape=(sequence_length, n_features),
                     name='bidirectional_lstm_1'),
        BatchNormalization(name='batch_norm_1'),
        Dropout(0.3, name='dropout_1'),
        
        # Second LSTM layer
        Bidirectional(LSTM(64), name='bidirectional_lstm_2'),
        BatchNormalization(name='batch_norm_2'),
        Dropout(0.3, name='dropout_2'),
        
        # Output layers
        Dense(32, activation='relu', name='dense_1'),
        Dense(1, activation='sigmoid', name='output')  # Binary output for lightning probability
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_and_visualize_model():
    """Train the model and create visualizations for thesis presentation"""
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Generate or load thesis data
    try:
        data = np.load(os.path.join(PROCESSED_DATA_DIR, 'thesis_training_sequences.npz'))
        X, y = data['X'], data['y']
        print(f"Loaded existing thesis data with {len(X)} samples")
        # Check if we have too few samples and regenerate if needed
        if len(X) < NUM_SAMPLES * 0.9:  # Allow for some flexibility
            print(f"Fewer samples than requested ({len(X)} < {NUM_SAMPLES}), regenerating...")
            X, y = generate_thesis_data(samples=NUM_SAMPLES)
    except:
        X, y = generate_thesis_data(samples=NUM_SAMPLES)
    
    # Split into training, validation, and test sets (70/15/15)
    # Using sklearn's train_test_split for proper stratification
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples, testing with {len(X_test)} samples")
    print(f"Training set: {np.sum(y_train)} positive examples ({np.sum(y_train)/len(y_train):.1%})")
    print(f"Validation set: {np.sum(y_val)} positive examples ({np.sum(y_val)/len(y_val):.1%})")
    print(f"Test set: {np.sum(y_test)} positive examples ({np.sum(y_test)/len(y_test):.1%})")
    
    # Build the model
    model = build_model(SEQUENCE_LENGTH, len(FEATURES))
    model.summary()
    
    # Try to save model architecture visualization, but continue if it fails
    try:
        model_image_path = os.path.join(MODEL_DIR, 'model_architecture.png')
        print("Attempting to create model architecture visualization...")
        # Using TensorFlow's plot_model function if available
        tf.keras.utils.plot_model(model, to_file=model_image_path, show_shapes=True, show_layer_names=True)
        print(f"Saved model architecture visualization to {model_image_path}")
    except Exception as e:
        print(f"Could not save model architecture visualization: {str(e)}")
        print("This is not critical - continuing with model training.")
        
        # Create a simple text file with the model architecture as an alternative
        with open(os.path.join(MODEL_DIR, 'model_architecture.txt'), 'w') as f:
            # Get the model summary as string
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Saved model architecture as text file instead.")
    
    # Calculate class weights to handle imbalance
    pos_weight = (len(y_train) / (2 * np.sum(y_train)))
    neg_weight = (len(y_train) / (2 * (len(y_train) - np.sum(y_train))))
    class_weights = {0: neg_weight, 1: pos_weight}
    print(f"Using class weights: {class_weights}")
    
    # Setup callbacks for saving best model
    checkpoint_path = os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5')
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,  # Increased batch size for faster training
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Save the final model
    model.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = os.path.join(MODEL_DIR, 'training_history.png')
    plt.savefig(history_path)
    plt.close()
    print(f"Saved training history visualization to {history_path}")
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[2]:.4f}")
    print(f"Test precision: {test_results[3]:.4f}")
    print(f"Test recall: {test_results[4]:.4f}")
    
    # Make predictions on test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Storm', 'Storm'], yticklabels=['No Storm', 'Storm'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    conf_matrix_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Saved confusion matrix to {conf_matrix_path}")
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(MODEL_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")
    
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=['No Storm', 'Storm'])
    with open(os.path.join(MODEL_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print(f"Saved classification report to {os.path.join(MODEL_DIR, 'classification_report.txt')}")
    
    # Visualize feature importance through permutation importance
    # Use a small subset for faster computation
    test_subset_size = min(1000, len(X_test))
    random_indices = np.random.choice(len(X_test), test_subset_size, replace=False)
    X_test_subset = X_test[random_indices]
    y_test_subset = y_test[random_indices]
    
    print(f"Computing feature importance using {test_subset_size} test samples...")
    feature_importance = {}
    baseline_score = model.evaluate(X_test_subset, y_test_subset, verbose=0)[1]  # baseline accuracy
    
    for i, feature_name in enumerate(FEATURES):
        # Create a copy of the test data
        X_test_permuted = X_test_subset.copy()
        
        # Permute the feature
        X_test_permuted[:, :, i] = np.random.permutation(X_test_permuted[:, :, i])
        
        # Evaluate with the permuted feature
        permuted_score = model.evaluate(X_test_permuted, y_test_subset, verbose=0)[1]
        
        # Importance is the drop in accuracy
        importance = baseline_score - permuted_score
        feature_importance[feature_name] = importance
        print(f"  {feature_name}: {importance:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    # Sort by importance
    sorted_idx = np.argsort(importances)
    plt.barh([features[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
    plt.xlabel('Decrease in Accuracy when Feature is Permuted')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    importance_path = os.path.join(MODEL_DIR, 'feature_importance.png')
    plt.savefig(importance_path)
    plt.close()
    print(f"Saved feature importance visualization to {importance_path}")
    
    # Save a summary of all metrics for easy reference
    with open(os.path.join(MODEL_DIR, 'model_summary.txt'), 'w') as f:
        f.write("Thunderstorm Prediction Model Summary\n")
        f.write("===================================\n\n")
        f.write("Model Architecture:\n")
        f.write("- Bidirectional LSTM (128 units) with sequence return\n")
        f.write("- BatchNormalization + Dropout (0.3)\n")
        f.write("- Bidirectional LSTM (64 units)\n") 
        f.write("- BatchNormalization + Dropout (0.3)\n")
        f.write("- Dense (32 units, ReLU activation)\n")
        f.write("- Output (1 unit, Sigmoid activation)\n\n")
        f.write(f"Total parameters: {model.count_params()}\n\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(f"Test loss: {test_results[0]:.4f}\n")
        f.write(f"Test accuracy: {test_results[1]:.4f}\n")
        f.write(f"Test AUC: {test_results[2]:.4f}\n")
        f.write(f"Test precision: {test_results[3]:.4f}\n")
        f.write(f"Test recall: {test_results[4]:.4f}\n\n")
        f.write("Feature importance:\n")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {feature}: {importance:.4f}\n")
    
    return model, history

def visualize_prediction_process():
    """Create a visualization of how the model makes predictions"""
    # Load or train the model
    model_path = os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5')
    
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model, _ = train_and_visualize_model()
    
    # Generate a few sample sequences (one with high storm probability, one with low)
    # We'll manually create these for demonstration
    X_samples = np.zeros((2, SEQUENCE_LENGTH, len(FEATURES)))
    
    # Sample 1: High storm probability
    # - High humidity
    # - Dropping pressure
    # - Increasing wind speed
    # - High cloud cover
    for t in range(SEQUENCE_LENGTH):
        # Temperature with daily cycle
        X_samples[0, t, 0] = 28 - 5 * np.sin(np.pi * t / 12)  # Cooler in afternoon (unusual)
        # Humidity increasing
        X_samples[0, t, 1] = 70 + t * 0.8
        # Pressure dropping
        X_samples[0, t, 2] = 1010 - t * 0.4
        # Wind speed increasing
        X_samples[0, t, 3] = 5 + t * 0.3
        # Wind direction shifting
        X_samples[0, t, 4] = (180 + t * 5) % 360
        # Precipitation increasing
        X_samples[0, t, 5] = max(0, t * 0.2 - 3) if t > 15 else 0
        # Cloud cover increasing
        X_samples[0, t, 6] = min(100, 50 + t * 2)
    
    # Sample 2: Low storm probability
    # - Moderate humidity
    # - Stable pressure
    # - Low wind speed
    # - Low cloud cover
    for t in range(SEQUENCE_LENGTH):
        # Normal temperature cycle
        X_samples[1, t, 0] = 30 + 5 * np.sin(np.pi * t / 12)  # Warmer in afternoon
        # Moderate humidity
        X_samples[1, t, 1] = 50 + 10 * np.sin(np.pi * t / 12)  # Slightly lower in afternoon
        # Stable pressure
        X_samples[1, t, 2] = 1015 + np.random.normal(0, 0.2)
        # Low wind speed
        X_samples[1, t, 3] = 3 + np.random.normal(0, 0.5)
        # Consistent wind direction
        X_samples[1, t, 4] = 90 + np.random.normal(0, 10)
        # No precipitation
        X_samples[1, t, 5] = 0
        # Low to moderate cloud cover
        X_samples[1, t, 6] = 30 + 10 * np.sin(np.pi * t / 12)  # Varying slightly
    
    # Make predictions
    predictions = model.predict(X_samples)
    
    # Create side-by-side plots showing input sequences and predictions
    fig, axes = plt.subplots(4, 2, figsize=(20, 16), sharex=True)
    plt.suptitle('Thunderstorm Prediction: Two Examples', fontsize=16)
    
    sample_labels = ['High Storm Probability', 'Low Storm Probability']
    pred_values = [f"{predictions[0][0]:.2%}", f"{predictions[1][0]:.2%}"]
    
    for col in range(2):
        # Add prediction value as title
        axes[0, col].set_title(f"{sample_labels[col]}\nPredicted Storm Probability: {pred_values[col]}", fontsize=14)
        
        # Temperature and humidity
        ax1 = axes[0, col]
        color = 'tab:red'
        ax1.set_ylabel('Temperature (°C)', color=color)
        ax1.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 0], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1_twin = ax1.twinx()
        color = 'tab:blue'
        ax1_twin.set_ylabel('Humidity (%)', color=color)
        ax1_twin.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 1], color=color, marker='s')
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Pressure
        ax2 = axes[1, col]
        ax2.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 2], color='purple', marker='o')
        ax2.set_ylabel('Pressure (hPa)')
        ax2.grid(True, alpha=0.3)
        
        # Wind
        ax3 = axes[2, col]
        color = 'tab:green'
        ax3.set_ylabel('Wind Speed (m/s)', color=color)
        ax3.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 3], color=color, marker='o')
        ax3.tick_params(axis='y', labelcolor=color)
        
        ax3_twin = ax3.twinx()
        color = 'tab:gray'
        ax3_twin.set_ylabel('Wind Direction (°)', color=color)
        ax3_twin.scatter(range(SEQUENCE_LENGTH), X_samples[col, :, 4], color=color, alpha=0.5)
        ax3_twin.tick_params(axis='y', labelcolor=color)
        ax3.grid(True, alpha=0.3)
        
        # Precipitation and cloud cover
        ax4 = axes[3, col]
        color = 'tab:blue'
        ax4.set_ylabel('Precipitation (mm)', color=color)
        ax4.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 5], color=color, marker='o')
        ax4.tick_params(axis='y', labelcolor=color)
        
        ax4_twin = ax4.twinx()
        color = 'tab:gray'
        ax4_twin.set_ylabel('Cloud Cover (%)', color=color)
        ax4_twin.plot(range(SEQUENCE_LENGTH), X_samples[col, :, 6], color=color, marker='s')
        ax4_twin.tick_params(axis='y', labelcolor=color)
        ax4.grid(True, alpha=0.3)
        
        # X axis label
        ax4.set_xlabel('Hour of Sequence', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    prediction_path = os.path.join(MODEL_DIR, 'prediction_examples.png')
    plt.savefig(prediction_path)
    plt.close()
    print(f"Saved prediction example visualization to {prediction_path}")

def create_thesis_visualization():
    """Create a comprehensive thesis visualization that explains the entire system"""
    plt.figure(figsize=(16, 12))
    
    # Create a flow diagram showing the system
    plt.subplot(1, 1, 1)
    
    # Define the positions
    positions = {
        'data': (0.1, 0.7),
        'preprocessing': (0.3, 0.7),
        'model': (0.5, 0.7),
        'prediction': (0.7, 0.7),
        'visualization': (0.9, 0.7),
        
        'temperature': (0.05, 0.9),
        'humidity': (0.1, 0.9),
        'pressure': (0.15, 0.9),
        'wind': (0.2, 0.9),
        'precip': (0.25, 0.9),
        'cloud': (0.3, 0.9),
        
        'data_clean': (0.3, 0.8),
        'data_sequence': (0.3, 0.6),
        
        'lstm1': (0.5, 0.8),
        'lstm2': (0.5, 0.7),
        'dense': (0.5, 0.6),
        'output': (0.5, 0.5),
        
        'prob': (0.7, 0.8),
        'thresh': (0.7, 0.6),
        
        'map': (0.9, 0.8),
        'alerts': (0.9, 0.6)
    }
    
    # Draw boxes
    for name, (x, y) in positions.items():
        # Main components (larger boxes)
        if name in ['data', 'preprocessing', 'model', 'prediction', 'visualization']:
            width, height = 0.12, 0.12
            rect = plt.Rectangle((x-width/2, y-height/2), width, height, fill=True, alpha=0.3, 
                                 color='lightblue', edgecolor='blue', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x, y, name.capitalize(), ha='center', va='center', fontsize=12, fontweight='bold')
        # Weather variables (small boxes at top)
        elif name in ['temperature', 'humidity', 'pressure', 'wind', 'precip', 'cloud']:
            width, height = 0.04, 0.04
            rect = plt.Rectangle((x-width/2, y-height/2), width, height, fill=True, alpha=0.3, 
                                 color='lightgreen', edgecolor='green', linewidth=1)
            plt.gca().add_patch(rect)
            short_name = name[:4]  # First 4 letters
            plt.text(x, y, short_name, ha='center', va='center', fontsize=8)
        # Model components
        elif name in ['lstm1', 'lstm2', 'dense', 'output']:
            width, height = 0.08, 0.03
            rect = plt.Rectangle((x-width/2, y-height/2), width, height, fill=True, alpha=0.3, 
                                 color='salmon', edgecolor='red', linewidth=1)
            plt.gca().add_patch(rect)
            layer_names = {'lstm1': 'BiLSTM 1', 'lstm2': 'BiLSTM 2', 'dense': 'Dense', 'output': 'Output'}
            plt.text(x, y, layer_names[name], ha='center', va='center', fontsize=8)
        # Other components
        else:
            width, height = 0.08, 0.03
            rect = plt.Rectangle((x-width/2, y-height/2), width, height, fill=True, alpha=0.3, 
                                 color='lightyellow', edgecolor='orange', linewidth=1)
            plt.gca().add_patch(rect)
            component_names = {
                'data_clean': 'Cleaning', 
                'data_sequence': 'Sequencing',
                'prob': 'Probability',
                'thresh': 'Thresholding',
                'map': 'Heatmap',
                'alerts': 'Alerts'
            }
            plt.text(x, y, component_names.get(name, name), ha='center', va='center', fontsize=8)
    
    # Draw arrows
    # Data sources to preprocessing
    for name in ['temperature', 'humidity', 'pressure', 'wind', 'precip', 'cloud']:
        x1, y1 = positions[name]
        x2, y2 = positions['data']
        plt.arrow(x1, y1-0.02, x2-x1, y2-y1+0.04, head_width=0.01, head_length=0.01, fc='gray', ec='gray')
    
    # Main flow
    for i in range(len(['data', 'preprocessing', 'model', 'prediction', 'visualization'])-1):
        start = ['data', 'preprocessing', 'model', 'prediction', 'visualization'][i]
        end = ['data', 'preprocessing', 'model', 'prediction', 'visualization'][i+1]
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        plt.arrow(x1+0.06, y1, x2-x1-0.12, 0, head_width=0.01, head_length=0.01, fc='blue', ec='blue')
    
    # Preprocessing details
    for name in ['data_clean', 'data_sequence']:
        x1, y1 = positions['preprocessing']
        x2, y2 = positions[name]
        if name == 'data_clean':
            plt.arrow(x1, y1+0.03, 0, y2-y1-0.05, head_width=0.01, head_length=0.01, fc='gray', ec='gray')
        else:
            plt.arrow(x1, y1-0.03, 0, y2-y1+0.05, head_width=0.01, head_length=0.01, fc='gray', ec='gray')
    
    # Model layers
    prev_layer = None
    for name in ['lstm1', 'lstm2', 'dense', 'output']:
        if prev_layer:
            x1, y1 = positions[prev_layer]
            x2, y2 = positions[name]
            plt.arrow(x1, y1-0.015, 0, y2-y1+0.03, head_width=0.01, head_length=0.01, fc='red', ec='red')
        prev_layer = name
    
    # Prediction details
    for name in ['prob', 'thresh']:
        x1, y1 = positions['prediction']
        x2, y2 = positions[name]
        if name == 'prob':
            plt.arrow(x1, y1+0.03, 0, y2-y1-0.05, head_width=0.01, head_length=0.01, fc='gray', ec='gray')
        else:
            plt.arrow(x1, y1-0.03, 0, y2-y1+0.05, head_width=0.01, head_length=0.01, fc='gray', ec='gray')
    
    # Visualization details
    for name in ['map', 'alerts']:
        x1, y1 = positions['visualization']
        x2, y2 = positions[name]
        if name == 'map':
            plt.arrow(x1, y1+0.03, 0, y2-y1-0.05, head_width=0.01, head_length=0.01, fc='gray', ec='gray')
        else:
            plt.arrow(x1, y1-0.03, 0, y2-y1+0.05, head_width=0.01, head_length=0.01, fc='gray', ec='gray')
    
    # Add title and adjust
    plt.title("Thunderstorm Prediction System Architecture", fontsize=18)
    plt.axis('off')  # Turn off the axis
    plt.xlim(0, 1)
    plt.ylim(0.4, 1)
    
    # Add explanatory text at bottom
    explanations = [
        "1. Data Collection: Weather variables gathered for Uttar Pradesh",
        "2. Preprocessing: Data cleaning and sequence creation (24-hour windows)",
        "3. Neural Network Model: Bidirectional LSTM network learns weather patterns",
        "4. Prediction: Model outputs probability of thunderstorm occurrence",
        "5. Visualization: Web interface displays storm risk across Uttar Pradesh"
    ]
    
    y_pos = 0.35
    for explanation in explanations:
        plt.text(0.1, y_pos, explanation, fontsize=10)
        y_pos -= 0.03
    
    plt.text(0.1, 0.2, "Key Features:", fontsize=12, fontweight='bold')
    key_features = [
        "• Deep learning approach identifies complex patterns humans might miss",
        "• Time-series analysis captures evolving weather conditions",
        "• Bidirectional processing considers both past and future trends",
        "• Multi-variable integration combines all relevant meteorological factors",
        "• Probability output provides nuanced risk assessment"
    ]
    
    y_pos = 0.18
    for feature in key_features:
        plt.text(0.1, y_pos, feature, fontsize=10)
        y_pos -= 0.03
    
    # Save the visualization
    system_path = os.path.join(MODEL_DIR, 'system_architecture.png')
    plt.savefig(system_path)
    plt.close()
    print(f"Saved system architecture visualization to {system_path}")

if __name__ == "__main__":
    # Create comprehensive visualizations for thesis presentation
    # First, generate visualizations that don't require model training
    visualize_sample_sequence()
    create_storm_vs_no_storm_comparison()
    create_thesis_visualization()
    
    # Then train model and create model-specific visualizations
    train_and_visualize_model()
    visualize_prediction_process()
    
    print("\nAll visualizations completed successfully!")
    print("\nFor your thesis presentation, use these files from the models/saved_models directory:")
    print("1. system_architecture.png - Overall system explanation")
    print("2. sample_sequence_visualization.png - Example of weather data input")
    print("3. storm_pattern_comparison.png - Comparison of storm vs. non-storm patterns")
    print("4. model_architecture.txt - Neural network architecture description")
    print("5. training_history.png - Model learning progress")
    print("6. confusion_matrix.png - Model performance summary")
    print("7. feature_importance.png - Which weather variables matter most")
    print("8. prediction_examples.png - How predictions are made")
    print("\nUse app.py with USE_MOCK_DATA=True for live demonstration")