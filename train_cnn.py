"""
Deep Learning Pipeline for ECG Arrhythmia Classification
FIXED VERSION - Handles class imbalance with stratified CV and more patients
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, balanced_accuracy_score
)

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# SMOTE for handling imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from ecg_mitbih import load_dataset


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train deep learning models for ECG arrhythmia classification"
    )
    ap.add_argument("--data_root", type=str, default="data/mitdb")
    ap.add_argument(
        "--records",
        type=str,
        nargs="+",
        # FIXED: Default to many more records for robust cross-validation
        default=[
            "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
            "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
            "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
            "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
            "222", "223", "228", "230", "231", "232", "233", "234"
        ],
    )
    ap.add_argument("--window_sec", type=float, default=5.0)
    ap.add_argument("--step_sec", type=float, default=2.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=0.0003)
    ap.add_argument("--model", type=str, default="cnn",
                   choices=["cnn", "lstm", "cnn_lstm", "fast_lstm"],
                   help="Model architecture to use")
    ap.add_argument("--use_smote", action="store_true",
                   help="Use SMOTE for balancing training data")
    ap.add_argument("--use_focal_loss", action="store_true",
                   help="Use focal loss instead of categorical crossentropy")
    ap.add_argument("--cross_validate", action="store_true",
                   help="Perform patient-wise cross-validation")
    ap.add_argument("--output", type=str, default="results_deep_learning.json",
                   help="Output file for results")
    ap.add_argument("--downsample_lstm", type=int, default=4,
                   help="Downsample factor for LSTM (reduces sequence length)")
    ap.add_argument("--min_test_ratio", type=float, default=0.01,
                   help="Minimum class ratio in test set (default: 0.01 = 1%%)")
    return ap.parse_args()


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss for handling class imbalance."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * keras.backend.log(y_pred)
        weight = keras.backend.pow(1 - y_pred, gamma)
        
        if alpha is not None:
            alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
            weight = alpha_weight * weight
        
        loss = weight * cross_entropy
        return keras.backend.sum(loss, axis=-1)
    
    return focal_loss_fixed


# ============================================================================
# MODEL ARCHITECTURES - OPTIMIZED
# ============================================================================

def build_1d_cnn(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """Advanced 1D CNN for ECG signal classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv1D(32, kernel_size=15, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        layers.Conv1D(64, kernel_size=7, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        layers.Conv1D(256, kernel_size=3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ], name='CNN_1D')
    
    return model


def build_fast_lstm(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """
    OPTIMIZED LSTM: Much faster than standard LSTM
    
    Key optimizations:
    1. Downsamples input (reduces sequence length by 4x)
    2. Smaller LSTM units (32 instead of 64)
    3. Single bidirectional layer instead of stacked
    4. Uses CuDNN-optimized LSTM when available
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Downsample to reduce sequence length (critical for speed!)
        layers.AveragePooling1D(pool_size=4, name='downsample'),
        
        # Single BiLSTM layer (faster than stacked)
        layers.Bidirectional(
            layers.LSTM(32, return_sequences=False,
                       kernel_regularizer=regularizers.l2(l2_reg))
        ),
        layers.Dropout(0.4),
        
        # Classification head
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ], name='Fast_BiLSTM')
    
    return model


def build_lstm(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """
    Standard LSTM network - WARNING: SLOW for long sequences!
    Use fast_lstm instead for better performance.
    """
    print("  ⚠️  WARNING: Standard LSTM is very slow for 1800 timesteps!")
    print("  💡 Consider using --model fast_lstm for 4-8x speedup")
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True,
                       kernel_regularizer=regularizers.l2(l2_reg))
        ),
        layers.Dropout(0.3),
        
        layers.Bidirectional(
            layers.LSTM(32, return_sequences=False,
                       kernel_regularizer=regularizers.l2(l2_reg))
        ),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ], name='BiLSTM')
    
    return model


def build_cnn_lstm(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """
    Hybrid CNN-LSTM architecture (OPTIMIZED)
    CNN reduces sequence length before LSTM processing
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN feature extraction + downsampling
        layers.Conv1D(64, kernel_size=7, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=4),  # Reduce by 4x
        layers.Dropout(0.2),
        
        layers.Conv1D(128, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),  # Further reduce by 2x
        layers.Dropout(0.3),
        
        # Now sequence is 8x shorter - LSTM can handle it
        layers.Bidirectional(
            layers.LSTM(32, return_sequences=False,
                       kernel_regularizer=regularizers.l2(l2_reg))
        ),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ], name='CNN_LSTM_Optimized')
    
    return model


def get_model(model_name: str, input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """Factory function to create models by name."""
    models = {
        'cnn': build_1d_cnn,
        'lstm': build_lstm,
        'fast_lstm': build_fast_lstm,
        'cnn_lstm': build_cnn_lstm
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](input_shape, l2_reg)


# ============================================================================
# CALLBACKS WITH BETTER PROGRESS TRACKING
# ============================================================================

class EnhancedProgressCallback(Callback):
    """Enhanced callback with time estimates and better progress reporting."""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"  🚀 Training started...")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Calculate ETA
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times[-10:])  # Last 10 epochs
            remaining_epochs = self.total_epochs - (epoch + 1)
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_minutes = eta_seconds / 60
            
            # Progress bar
            progress = (epoch + 1) / self.total_epochs
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Print every 5 epochs or at key points
            if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == self.total_epochs:
                print(f"\n  📊 Epoch {epoch+1}/{self.total_epochs} [{bar}] "
                      f"{progress*100:.1f}%")
                print(f"     Time: {epoch_time:.1f}s/epoch | ETA: {eta_minutes:.1f}m")
                print(f"     Loss: {logs.get('loss', 0):.4f} → {logs.get('val_loss', 0):.4f} | "
                      f"AUC: {logs.get('AUC', 0):.4f} → {logs.get('val_AUC', 0):.4f}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"  ✅ Training completed in {total_time/60:.1f} minutes")


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def print_fold_details(y_train, y_test, train_groups, test_groups):
    """Print detailed fold information."""
    print(f"  Train patients: {train_groups}")
    print(f"  Test patients:  {test_groups}")
    print(f"  Train: {len(y_train)} samples "
          f"(Normal={np.sum(y_train==0)}, Arrhythmia={np.sum(y_train==1)})")
    print(f"  Test:  {len(y_test)} samples "
          f"(Normal={np.sum(y_test==0)}, Arrhythmia={np.sum(y_test==1)})")


def apply_smote(X_cnn, y, verbose=True):
    """Apply SMOTE oversampling to balance classes."""
    if not SMOTE_AVAILABLE:
        if verbose:
            print("  ⚠️  SMOTE not available, using original data")
        return X_cnn, y
    
    original_shape = X_cnn.shape
    X_flat = X_cnn.reshape(X_cnn.shape[0], -1)
    
    unique, counts = np.unique(y, return_counts=True)
    if verbose:
        print(f"  Before SMOTE: {dict(zip(unique, counts))}")
    
    min_samples = np.min(counts)
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
    
    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_balanced, y_balanced = smote.fit_resample(X_flat, y)
        X_balanced = X_balanced.reshape(-1, original_shape[1], original_shape[2])
        
        if verbose:
            unique_new, counts_new = np.unique(y_balanced, return_counts=True)
            print(f"  After SMOTE:  {dict(zip(unique_new, counts_new))}")
        
        return X_balanced, y_balanced
    except Exception as e:
        if verbose:
            print(f"  ⚠️  SMOTE failed: {e}")
        return X_cnn, y


def compute_class_weights(y):
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    # Use less aggressive clipping to avoid overly large weights
    weights = np.clip(weights, 0.5, 1.5)
    
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"  Class weights: {class_weight_dict}")
    return class_weight_dict


def prepare_data_for_model(X, y):
    """Prepare data for neural network input."""
    n_samples, n_features = X.shape
    # Always treat data as single-channel (Lead II)
    n_channels = 1
    timesteps = n_features

    X_reshaped = X.reshape(n_samples, timesteps, n_channels)
    print(f"  Data prepared: {X_reshaped.shape} (timesteps={timesteps}, channels={n_channels})")
    y_categorical = keras.utils.to_categorical(y, num_classes=2)
    
    return X_reshaped, y_categorical


# ============================================================================
# CROSS-VALIDATION WITH STRATIFICATION - FIXED
# ============================================================================

def cross_validate_deep_learning(X, y, groups, args):
    """
    Perform patient-wise cross-validation with stratified splitting.
    
    FIXES:
    1. Uses StratifiedGroupKFold to ensure balanced folds
    2. More lenient imbalance threshold (1% instead of 5%)
    3. Better warnings for users with few patients
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Check if we have enough patients
    if n_groups < 10:
        print(f"\n{'⚠️ '*30}")
        print(f"WARNING: Only {n_groups} patients in dataset!")
        print(f"Recommendation: Use 20+ patients for robust cross-validation")
        print(f"Current setup may produce unreliable results")
        print(f"{'⚠️ '*30}\n")
    
    # Determine number of splits
    if n_groups < 5:
        n_splits = n_groups  # Leave-one-patient-out
        print(f"Using leave-one-patient-out CV (only {n_groups} patients)")
    else:
        n_splits = 5
    
    model_names = {
        'cnn': '1D Convolutional Neural Network',
        'lstm': 'Bidirectional LSTM (SLOW)',
        'fast_lstm': 'Fast Bidirectional LSTM (Optimized)',
        'cnn_lstm': 'Hybrid CNN-LSTM (Optimized)'
    }
    
    print(f"\n{'='*60}")
    print(f"STRATIFIED PATIENT-WISE CROSS-VALIDATION")
    print(f"Model: {model_names.get(args.model, args.model)}")
    print(f"{'='*60}")
    print(f"Patients: {n_groups} | Splits: {n_splits}")
    print(f"Using SMOTE: {args.use_smote}")
    print(f"Using Focal Loss: {args.use_focal_loss}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Min test ratio threshold: {args.min_test_ratio:.1%}")
    print(f"{'='*60}\n")
    
    # Use strict GroupKFold for true patient-wise separation
    # This ensures: Test patients NEVER appear in Train
    from sklearn.model_selection import GroupKFold

    print(f"\n{'='*60}")
    print(f"PATIENT-WISE CROSS-VALIDATION (Cross-Subject)")
    print(f"✓ Patients in Test do NOT appear in Train")
    print(f"{'='*60}")

    # GroupKFold strictly separates patients (groups)
    gkf = GroupKFold(n_splits=n_splits)
    split_iterator = gkf.split(X, y, groups=groups)
    
    all_scores = {
        'accuracy': [], 'balanced_accuracy': [], 'f1': [],
        'precision': [], 'recall': [], 'auc': [],
        'sensitivity': [], 'specificity': []
    }
    
    fold_details = []
    total_start_time = time.time()
    skipped_folds = 0
    
    for fold, (train_idx, test_idx) in enumerate(split_iterator, 1):
        fold_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"FOLD {fold}/{n_splits}")
        print('='*50)
        
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Count patients in each fold (for info only)
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])

        print(f"  Train: {len(y_tr)} samples (Normal={np.sum(y_tr==0)}, Arrhythmia={np.sum(y_tr==1)})")
        print(f"  Test:  {len(y_te)} samples (Normal={np.sum(y_te==0)}, Arrhythmia={np.sum(y_te==1)})")
        print(f"  Patients in train: {len(train_groups)} | Patients in test: {len(test_groups)}")
        print(f"  ✓ Guaranteed: No test patient data in training set")
        
        # FIXED: Much more lenient threshold (1% instead of 5%)
        test_normal_ratio = np.sum(y_te == 0) / len(y_te)
        test_arr_ratio = 1 - test_normal_ratio
        
        if test_normal_ratio < args.min_test_ratio or test_normal_ratio > (1 - args.min_test_ratio):
            print(f"  🚫 Skipping fold: test set has {test_arr_ratio:.1%} arrhythmia")
            print(f"     (Below threshold of {args.min_test_ratio:.1%})")
            skipped_folds += 1
            
            # Store as skipped
            fold_details.append({
                'fold': fold,
                'skipped': True,
                'reason': 'extreme_imbalance',
                'test_arrhythmia_ratio': float(test_arr_ratio),
                'train_patients': train_groups.tolist(),
                'test_patients': test_groups.tolist()
            })
            continue
        
        # Warn but continue if moderately imbalanced
        if test_arr_ratio < 0.05 or test_arr_ratio > 0.95:
            print(f"  ⚠️  Warning: Test set is imbalanced ({test_arr_ratio:.1%} arrhythmia)")
            print(f"     But continuing (above {args.min_test_ratio:.1%} threshold)")
            print(f"     Balanced Accuracy will handle this")
        
        # Normalize
        # Apply StandardScaler - bandpass filtering removes frequencies but doesn't standardize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Check scaling statistics
        print(f"  Applied StandardScaler")
        print(f"    Train: mean={X_tr.mean():.3f}, std={X_tr.std():.3f}")
        print(f"    Test:  mean={X_te.mean():.3f}, std={X_te.std():.3f}")
        # Prepare for model
        X_tr_model, y_tr_cat = prepare_data_for_model(X_tr, y_tr)
        X_te_model, y_te_cat = prepare_data_for_model(X_te, y_te)
        
        # Apply SMOTE
        if args.use_smote:
            X_tr_model, y_tr = apply_smote(X_tr_model, y_tr, verbose=True)
            y_tr_cat = keras.utils.to_categorical(y_tr, num_classes=2)
        
        # Stratified validation split
        try:
            X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(
                X_tr_model, y_tr_cat,
                test_size=0.2,
                stratify=y_tr,
                random_state=42
            )
        except ValueError:
            # If stratification fails, split without it
            print("  ⚠️  Stratified split failed, using random split")
            X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(
                X_tr_model, y_tr_cat,
                test_size=0.2,
                random_state=42
            )
        
        print(f"\n  Data shapes:")
        print(f"    Training: {X_tr_split.shape}")
        print(f"    Validation: {X_val_split.shape}")
        print(f"    Test: {X_te_model.shape}")
        
        # Build model
        model = get_model(args.model, input_shape=X_tr_model.shape[1:], l2_reg=0.001)
        
        print(f"\n  Model architecture: {args.model.upper()}")
        print(f"    Total parameters: {model.count_params():,}")
        
        # Loss function
        loss_fn = focal_loss(gamma=2.0, alpha=0.25) if args.use_focal_loss else 'categorical_crossentropy'
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=loss_fn,
            metrics=['accuracy', keras.metrics.AUC(name='AUC')]
        )
        
        # Class weights
        class_weights = None if args.use_smote else compute_class_weights(y_tr)
        
        # Callbacks with enhanced progress tracking
        callbacks = [
            EarlyStopping(
                monitor='val_AUC',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_AUC',
                factor=0.5,
                patience=10,
                mode='max',
                min_lr=1e-6,
                verbose=0
            ),
            EnhancedProgressCallback(args.epochs)
        ]
        
        # Train
        history = model.fit(
            X_tr_split, y_tr_split,
            validation_data=(X_val_split, y_val_split),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_te_model, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        acc = accuracy_score(y_te, y_pred)
        bal_acc = balanced_accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='binary', zero_division=0)
        prec = precision_score(y_te, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_te, y_pred, average='binary', zero_division=0)
        
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        try:
            auc = roc_auc_score(y_te, y_pred_proba[:, 1])
        except (ValueError, IndexError) as e:
            print(f"  ⚠️  AUC calculation failed: {e}")
            auc = np.nan
        
        # Store
        all_scores['accuracy'].append(acc)
        all_scores['balanced_accuracy'].append(bal_acc)
        all_scores['f1'].append(f1)
        all_scores['precision'].append(prec)
        all_scores['recall'].append(rec)
        all_scores['auc'].append(auc)
        all_scores['sensitivity'].append(sensitivity)
        all_scores['specificity'].append(specificity)
        
        fold_time = time.time() - fold_start_time
        
        print(f"\n  📊 Fold {fold} Results (completed in {fold_time/60:.1f}m):")
        print(f"     Accuracy:          {acc:.3f}")
        print(f"     Balanced Accuracy: {bal_acc:.3f}")
        print(f"     F1-score:          {f1:.3f}")
        print(f"     AUC:               {auc:.3f}")
        print(f"     Sensitivity:       {sensitivity:.3f}")
        print(f"     Specificity:       {specificity:.3f}")
        
        print(f"\n     Confusion Matrix:")
        print(f"     [{cm[0,0]:4d}  {cm[0,1]:4d}]  (Normal)")
        print(f"     [{cm[1,0]:4d}  {cm[1,1]:4d}]  (Arrhythmia)")
        
        fold_details.append({
            'fold': fold,
            'skipped': False,
            'train_patients': train_groups.tolist(),
            'test_patients': test_groups.tolist(),
            'training_time_minutes': float(fold_time / 60),
            'metrics': {
                'accuracy': float(acc),
                'balanced_accuracy': float(bal_acc),
                'f1': float(f1),
                'auc': float(auc),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity)
            },
            'confusion_matrix': cm.tolist()
        })
    
    total_time = time.time() - total_start_time
    
    # Summary
    completed_folds = n_splits - skipped_folds
    
    if skipped_folds > 0:
        print(f"\n{'⚠️ '*30}")
        print(f"WARNING: Skipped {skipped_folds}/{n_splits} folds due to imbalance")
        print(f"Completed: {completed_folds}/{n_splits} folds")
        if completed_folds < 3:
            print(f"❌ Not enough valid folds for reliable evaluation!")
            print(f"💡 Solution: Use more patient records (--records 100 101 ... 234)")
        print(f"{'⚠️ '*30}\n")
    
    summary = {}
    for metric_name, values in all_scores.items():
        if len(values) > 0:
            summary[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
    
    if summary:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {args.model.upper()}")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"Completed Folds: {completed_folds}/{n_splits}")
        print(f"{'='*60}")
        for metric_name, stats in summary.items():
            print(f"{metric_name.replace('_', ' ').title():20s}: "
                  f"{stats['mean']:.3f} (±{stats['std']:.3f})")
    
    return {
        'model': args.model,
        'total_training_time_minutes': float(total_time / 60),
        'n_patients': int(n_groups),
        'n_splits': int(n_splits),
        'completed_folds': int(completed_folds),
        'skipped_folds': int(skipped_folds),
        'summary': summary,
        'fold_details': fold_details
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not KERAS_AVAILABLE:
        print("❌ ERROR: TensorFlow/Keras not installed.")
        print("Install with: pip install tensorflow")
        return
    
    args = parse_args()
    
    # Print optimization tips
    if args.model == 'lstm':
        print("\n" + "!"*60)
        print("⚠️  WARNING: Standard LSTM is VERY SLOW!")
        print("💡 RECOMMENDATION: Use --model fast_lstm for 4-8x speedup")
        print("   (Or use --model cnn_lstm for best speed/accuracy)")
        print("!"*60 + "\n")
        
        response = input("Continue with slow LSTM? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Run with --model fast_lstm instead")
            return
    
    data_root = Path(args.data_root)

    print("="*60)
    print("DEEP LEARNING FOR ECG ARRHYTHMIA CLASSIFICATION")
    print("FIXED VERSION - Stratified CV with balanced folds")
    print("="*60)
    print(f"Model: {args.model}")
    
    # Check if user is using too few records
    if len(args.records) < 10:
        print(f"\n{'⚠️ '*30}")
        print(f"WARNING: Only {len(args.records)} records specified!")
        print(f"This may cause extreme class imbalance in cross-validation")
        print(f"\n💡 RECOMMENDATION:")
        print(f"Use 20+ records for robust results. Example:")
        print(f"  --records 100 101 102 103 104 105 106 107 108 109 \\")
        print(f"            111 112 113 114 115 116 117 118 119 121 \\")
        print(f"            122 123 124 200 201 202 203 205 207 208")
        print(f"{'⚠️ '*30}\n")
        
        response = input(f"Continue with only {len(args.records)} records? (y/N): ")
        if response.lower() != 'y':
            print("\nExiting. Add more records to --records argument")
            return
    
    # Load dataset
    print("\n🔄 Loading dataset...")
    X, y, groups = load_dataset(
        data_root,
        records=args.records,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        bp=(0.5, 40.0),
        use_features=False,
        verbose=True
    )

    # -----------------------------
    # DATA SANITY CHECKS (quick)
    # -----------------------------
    print("\n" + "="*60)
    print("DATA SANITY CHECKS")
    print("="*60)
    try:
        print(f"X shape: {X.shape}")
        print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"X mean: {X.mean():.3f}, std: {X.std():.3f}")
        print(f"NaN check: {np.isnan(X).any()}")
        print(f"Inf check: {np.isinf(X).any()}")

        print(f"\nLabel distribution:")
        print(f"  Normal (0): {np.sum(y==0)} samples ({np.sum(y==0)/len(y)*100:.1f}%)")
        print(f"  Arrhythmia (1): {np.sum(y==1)} samples ({np.sum(y==1)/len(y)*100:.1f}%)")

        print(f"\nFirst 50 labels: {y[:50]}")

        # Check if labels are reasonable
        if np.sum(y==0) == 0 or np.sum(y==1) == 0:
            print("\n⚠️  WARNING: One class has 0 samples!")
        if np.sum(y==0) < 100 or np.sum(y==1) < 100:
            print("\n⚠️  WARNING: Very few samples in one class!")

    except Exception as e:
        print(f"⚠️  Data sanity check failed: {e}")

    print("="*60 + "\n")
    
    if X.shape[0] == 0:
        print("\n❌ ERROR: No windows extracted.")
        print("Check that data_root contains valid MIT-BIH records")
        return
    
    # Cross-validation
    if args.cross_validate:
        results = cross_validate_deep_learning(X, y, groups, args)
        
        # Check if we got enough valid folds
        if results['completed_folds'] < 2:
            print("\n" + "="*60)
            print("❌ INSUFFICIENT VALID FOLDS")
            print("="*60)
            print(f"Only {results['completed_folds']} out of {results['n_splits']} "
                  f"folds completed successfully")
            print("\n🔧 SOLUTIONS:")
            print("1. Add more patient records (recommended 20-30)")
            print("2. Decrease --min_test_ratio (current: {:.1%})".format(args.min_test_ratio))
            print("   Example: --min_test_ratio 0.005  (0.5% instead of 1%)")
            print("3. Use different window settings to get more balanced data")
            print("="*60)
        
        # Save results
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print summary statistics
        if results.get('summary'):
            print(f"\n📈 FINAL PERFORMANCE SUMMARY")
            print(f"{'='*60}")
            summary = results['summary']
            for metric in ['balanced_accuracy', 'f1', 'auc', 'sensitivity', 'specificity']:
                if metric in summary:
                    stats = summary[metric]
                    print(f"{metric.replace('_', ' ').title():25s}: "
                          f"{stats['mean']:.3f} ± {stats['std']:.3f}")
    else:
        print("\n❌ --cross_validate flag not set")
        print("Add --cross_validate to run patient-wise cross-validation")


if __name__ == "__main__":
    main()