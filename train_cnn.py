"""
Deep Learning Pipeline for ECG Arrhythmia Classification
UPDATED VERSION - Uses Binary Crossentropy for true binary classification
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple
import time

import numpy as np
from sklearn.model_selection import train_test_split
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
        default=["100", "101", "102", "103", "104", "105", "106", "107", "108", "109"],
    )
    ap.add_argument("--window_sec", type=float, default=5.0)
    ap.add_argument("--step_sec", type=float, default=2.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=0.0001)
    ap.add_argument("--model", type=str, default="cnn",
                   choices=["cnn", "lstm", "cnn_lstm", "fast_lstm"],
                   help="Model architecture to use")
    ap.add_argument("--use_smote", action="store_true",
                   help="Use SMOTE for balancing training data")
    ap.add_argument("--cross_validate", action="store_true",
                   help="Perform patient-wise cross-validation")
    ap.add_argument("--output", type=str, default="results_deep_learning.json",
                   help="Output file for results")
    ap.add_argument("--min_test_ratio", type=float, default=0.01,
                   help="Minimum class ratio in test set (default: 0.01 = 1%%)")
    return ap.parse_args()


# ============================================================================
# MODEL ARCHITECTURES - WITH BINARY OUTPUT
# ============================================================================

def build_1d_cnn(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """
    Advanced 1D CNN for ECG signal classification.
    ✅ UPDATED: Uses sigmoid output for true binary classification
    """
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
        
        # ✅ CHANGED: Binary output with sigmoid
        layers.Dense(1, activation='sigmoid')  # Was: Dense(2, softmax)
    ], name='CNN_1D')
    
    return model


def build_fast_lstm(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """
    OPTIMIZED LSTM: Much faster than standard LSTM
    ✅ UPDATED: Uses sigmoid output for true binary classification
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
        
        # ✅ CHANGED: Binary output with sigmoid
        layers.Dense(1, activation='sigmoid')  # Was: Dense(2, softmax)
    ], name='Fast_BiLSTM')
    
    return model


def build_lstm(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """
    Standard LSTM network - WARNING: SLOW for long sequences!
    ✅ UPDATED: Uses sigmoid output for true binary classification
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
        
        # ✅ CHANGED: Binary output with sigmoid
        layers.Dense(1, activation='sigmoid')  # Was: Dense(2, softmax)
    ], name='BiLSTM')
    
    return model


def build_cnn_lstm(input_shape: Tuple[int, int], l2_reg: float = 0.001):
    """
    Hybrid CNN-LSTM architecture (OPTIMIZED)
    ✅ UPDATED: Uses sigmoid output for true binary classification
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN feature extraction + downsampling
        layers.Conv1D(64, kernel_size=7, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=4),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(pool_size=2),
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
        
        # ✅ CHANGED: Binary output with sigmoid
        layers.Dense(1, activation='sigmoid')  # Was: Dense(2, softmax)
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
            avg_epoch_time = np.mean(self.epoch_times[-10:])
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
                # ✅ UPDATED: Show binary_crossentropy loss
                print(f"     Loss: {logs.get('loss', 0):.4f} → {logs.get('val_loss', 0):.4f} | "
                      f"AUC: {logs.get('AUC', 0):.4f} → {logs.get('val_AUC', 0):.4f}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"  ✅ Training completed in {total_time/60:.1f} minutes")


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

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
    weights = np.clip(weights, 0.5, 1.5)
    
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"  Class weights: {class_weight_dict}")
    return class_weight_dict


def prepare_data_for_model(X, y, binary_output=True):
    """
    Prepare data for neural network input.
    ✅ UPDATED: Supports both binary and categorical outputs
    """
    n_samples, n_features = X.shape
    n_channels = 1
    timesteps = n_features

    X_reshaped = X.reshape(n_samples, timesteps, n_channels)
    print(f"  Data prepared: {X_reshaped.shape} (timesteps={timesteps}, channels={n_channels})")
    
    # ✅ CHANGED: Binary labels (0, 1) instead of categorical
    if binary_output:
        y_output = y.astype(np.float32)  # Just 0.0 and 1.0
        print(f"  Labels: Binary format (0.0, 1.0)")
    else:
        y_output = keras.utils.to_categorical(y, num_classes=2)
        print(f"  Labels: Categorical format ([1,0], [0,1])")
    
    return X_reshaped, y_output


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validate_deep_learning(X, y, groups, args):
    """
    Perform patient-wise cross-validation.
    ✅ UPDATED: Uses binary crossentropy
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    if n_groups < 10:
        print(f"\n{'⚠️ '*30}")
        print(f"WARNING: Only {n_groups} patients in dataset!")
        print(f"{'⚠️ '*30}\n")
    
    n_splits = n_groups if n_groups < 5 else 5
    
    model_names = {
        'cnn': '1D Convolutional Neural Network',
        'lstm': 'Bidirectional LSTM (SLOW)',
        'fast_lstm': 'Fast Bidirectional LSTM (Optimized)',
        'cnn_lstm': 'Hybrid CNN-LSTM (Optimized)'
    }
    
    print(f"\n{'='*60}")
    print(f"PATIENT-WISE CROSS-VALIDATION (Binary Classification)")
    print(f"Model: {model_names.get(args.model, args.model)}")
    print(f"{'='*60}")
    print(f"Patients: {n_groups} | Splits: {n_splits}")
    print(f"Using SMOTE: {args.use_smote}")
    print(f"Loss: Binary Crossentropy")  # ✅ NEW
    print(f"Output: Sigmoid (single neuron)")  # ✅ NEW
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    from sklearn.model_selection import GroupKFold
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
        
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])

        print(f"  Train: {len(y_tr)} samples (Normal={np.sum(y_tr==0)}, Arrhythmia={np.sum(y_tr==1)})")
        print(f"  Test:  {len(y_te)} samples (Normal={np.sum(y_te==0)}, Arrhythmia={np.sum(y_te==1)})")
        print(f"  ✓ Guaranteed: No test patient data in training set")
        
        # Check imbalance
        test_normal_ratio = np.sum(y_te == 0) / len(y_te)
        test_arr_ratio = 1 - test_normal_ratio
        
        if test_normal_ratio < args.min_test_ratio or test_normal_ratio > (1 - args.min_test_ratio):
            print(f"  🚫 Skipping fold: test set has {test_arr_ratio:.1%} arrhythmia")
            skipped_folds += 1
            fold_details.append({
                'fold': fold,
                'skipped': True,
                'reason': 'extreme_imbalance'
            })
            continue
        
        if test_arr_ratio < 0.05 or test_arr_ratio > 0.95:
            print(f"  ⚠️  Warning: Test set is imbalanced ({test_arr_ratio:.1%} arrhythmia)")
        
        # Normalize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        print(f"  Applied StandardScaler")
        print(f"    Train: mean={X_tr.mean():.3f}, std={X_tr.std():.3f}")
        
        # ✅ CHANGED: Binary output instead of categorical
        X_tr_model, y_tr_binary = prepare_data_for_model(X_tr, y_tr, binary_output=True)
        X_te_model, y_te_binary = prepare_data_for_model(X_te, y_te, binary_output=True)
        
        # Apply SMOTE
        if args.use_smote:
            X_tr_model, y_tr = apply_smote(X_tr_model, y_tr, verbose=True)
            y_tr_binary = y_tr.astype(np.float32)
        
        # Stratified validation split
        try:
            X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(
                X_tr_model, y_tr_binary,
                test_size=0.2,
                stratify=y_tr if args.use_smote else y_tr_binary.astype(int),
                random_state=42
            )
        except ValueError:
            X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(
                X_tr_model, y_tr_binary,
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
        print(f"    Output: 1 neuron with sigmoid activation")
        print(f"    Loss: Binary Crossentropy")
        
        # ✅ CHANGED: Binary crossentropy instead of categorical
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='binary_crossentropy',  # Was: 'categorical_crossentropy'
            metrics=['accuracy', keras.metrics.AUC(name='AUC')]
        )
        
        # Class weights
        class_weights = None if args.use_smote else compute_class_weights(y_tr)
        
        # Callbacks
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
        
        # ✅ CHANGED: Predict with sigmoid output
        y_pred_proba = model.predict(X_te_model, verbose=0).flatten()  # Shape: (n_samples,)
        y_pred = (y_pred_proba > 0.5).astype(int)  # Threshold at 0.5
        
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
            auc = roc_auc_score(y_te, y_pred_proba)
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
    completed_folds = n_splits - skipped_folds
    
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
        'loss_function': 'binary_crossentropy',
        'output_activation': 'sigmoid',
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
    
    if args.model == 'lstm':
        print("\n" + "!"*60)
        print("⚠️  WARNING: Standard LSTM is VERY SLOW!")
        print("💡 RECOMMENDATION: Use --model fast_lstm for 4-8x speedup")
        print("!"*60 + "\n")
        
        response = input("Continue with slow LSTM? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Run with --model fast_lstm instead")
            return
    
    data_root = Path(args.data_root)

    print("="*60)
    print("DEEP LEARNING FOR ECG ARRHYTHMIA CLASSIFICATION")
    print("UPDATED: Binary Crossentropy + Sigmoid Output")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Loss: Binary Crossentropy")
    print(f"Output: Sigmoid (1 neuron)")
    
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
    
    if X.shape[0] == 0:
        print("\n❌ ERROR: No windows extracted.")
        return
    
    # Cross-validation
    if args.cross_validate:
        results = cross_validate_deep_learning(X, y, groups, args)
        
        # Save results
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
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


if __name__ == "__main__":
    main()
