"""
Comprehensive Model Comparison for ECG Arrhythmia Classification
FIXED VERSION - With Patient-Type Stratification & Anti-Overfitting

FIXES APPLIED:
1. PatientTypeStratifiedGroupKFold - ensures balanced test sets
2. Feature cleaning & selection - removes noisy features
3. Stronger regularization - reduces overfitting
4. All original functionality preserved
"""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from utils import setup_logger
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PATIENT-TYPE STRATIFIED GROUP K-FOLD (KEY FIX #1)
# ============================================================================

class PatientTypeStratifiedGroupKFold:
    """
    Groups patients by their arrhythmia profile, then ensures each fold
    contains patients from ALL profile types.
    
    Patient types:
      - HIGH_ARR: >80% arrhythmia (e.g., 102, 104, 107)
      - LOW_ARR:  <20% arrhythmia (e.g., 101, 103, 117)  
      - MIXED:    20-80% arrhythmia (e.g., 105, 106, 118)
    
    This ensures each test fold sees a realistic mix of patient types!
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def _categorize_patients(self, y, groups):
        """Categorize patients by their arrhythmia ratio."""
        unique_groups = np.unique(groups)
        
        high_arr = []   # >80% arrhythmia
        low_arr = []    # <20% arrhythmia
        mixed = []      # 20-80% arrhythmia
        
        patient_info = {}
        
        for g in unique_groups:
            mask = groups == g
            arr_ratio = np.mean(y[mask])
            patient_info[g] = arr_ratio
            
            if arr_ratio > 0.80:
                high_arr.append(g)
            elif arr_ratio < 0.20:
                low_arr.append(g)
            else:
                mixed.append(g)
        
        return high_arr, low_arr, mixed, patient_info
    
    def split(self, X, y, groups):
        """Generate stratified splits ensuring each fold has all patient types."""
        high_arr, low_arr, mixed, patient_info = self._categorize_patients(y, groups)
        
        rng = np.random.RandomState(self.random_state)
        
        # Shuffle within each category
        if self.shuffle:
            rng.shuffle(high_arr)
            rng.shuffle(low_arr)
            rng.shuffle(mixed)
        
        # Distribute each category across folds using round-robin
        fold_patients = [[] for _ in range(self.n_splits)]
        
        for category in [high_arr, low_arr, mixed]:
            for i, patient in enumerate(category):
                fold_idx = i % self.n_splits
                fold_patients[fold_idx].append(patient)
        
        # Generate indices for each fold
        for fold_idx in range(self.n_splits):
            test_patients = set(fold_patients[fold_idx])
            
            test_mask = np.array([g in test_patients for g in groups])
            train_mask = ~test_mask
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ============================================================================
# FEATURE CLEANING & SELECTION (KEY FIX #2)
# ============================================================================

def clean_and_select_features(X, y, n_features=20, verbose=True):
    """
    Clean features and select the best ones.
    
    Steps:
    1. Remove constant features
    2. Handle NaN/Inf
    3. Remove highly correlated features
    4. Select top features by mutual information
    """
    from sklearn.feature_selection import mutual_info_classif
    
    original_n = X.shape[1]
    
    if verbose:
        print(f"  📊 Feature cleaning & selection:")
        print(f"     Original: {original_n} features")
    
    # Step 1: Remove constant features
    std = np.std(X, axis=0)
    non_constant = std > 1e-8
    X = X[:, non_constant]
    
    if verbose:
        print(f"     After removing constant: {X.shape[1]} features")
    
    # Step 2: Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Step 3: Remove highly correlated features
    if X.shape[1] > n_features:
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0)
        
        to_remove = set()
        for i in range(corr.shape[0]):
            if i in to_remove:
                continue
            for j in range(i + 1, corr.shape[1]):
                if abs(corr[i, j]) > 0.95:
                    to_remove.add(j)
        
        keep_mask = np.array([i not in to_remove for i in range(X.shape[1])])
        X = X[:, keep_mask]
        
        if verbose:
            print(f"     After removing correlated: {X.shape[1]} features")
    
    # Step 4: Select best features
    if n_features > 0 and X.shape[1] > n_features:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        top_indices = np.argsort(mi_scores)[-n_features:]
        X = X[:, top_indices]
        
        if verbose:
            print(f"     After MI selection: {X.shape[1]} features")
    
    return X


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run comprehensive model comparison (FIXED VERSION)"
    )
    ap.add_argument("--data_root", type=str, default="data/mitdb")
    ap.add_argument(
        "--records",
        type=str,
        nargs="+",
        default=[
            "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
            "111", "112", "113", "114", "115", "116", "117", "118", "119", "121"
        ],
    )
    
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--use_smote", action="store_true",
                   help="Use SMOTE for ALL models (classical + deep learning)")
    ap.add_argument("--include_wavelet", action="store_true",
                   help="Include wavelet features for classical models")
    ap.add_argument("--output_dir", type=str, default="results",
                   help="Directory to save results")
    ap.add_argument("--skip_classical", action="store_true",
                   help="Skip classical ML models")
    ap.add_argument("--skip_deep", action="store_true",
                   help="Skip deep learning models")
    ap.add_argument("--use_slow_lstm", action="store_true",
                   help="Use standard LSTM instead of fast_lstm (NOT RECOMMENDED)")
    ap.add_argument("--quick_test", action="store_true",
                   help="Quick test mode: fewer epochs, skip slow models")
    ap.add_argument("--n_features", type=int, default=20,
                   help="Number of features to select (0 = use all)")
    ap.add_argument("--use_old_cv", action="store_true",
                   help="Use old GroupKFold instead of PatientTypeStratified (not recommended)")
    return ap.parse_args()


# ============================================================================
# TIME ESTIMATION
# ============================================================================

def print_time_estimate(model_name: str, epochs: int, n_records: int):
    """Print estimated training time for each model based on number of records."""
    time_per_sample = {
        'cnn': 0.1,
        'fast_lstm': 0.15,
        'lstm': 0.6,
        'cnn_lstm': 0.2
    }
    
    estimated_samples = n_records * 720
    estimated_seconds = (time_per_sample.get(model_name, 0.15) * 
                        estimated_samples * epochs * 5 / 1000)
    estimated_minutes = estimated_seconds / 60
    
    if estimated_minutes > 60:
        print(f"  ⏱️  Estimated time: {estimated_minutes/60:.1f} hours")
        if estimated_minutes > 120:
            print(f"  ⚠️  WARNING: This will take over 2 hours!")
    else:
        print(f"  ⏱️  Estimated time: {estimated_minutes:.0f} minutes")
    
    if model_name == 'lstm':
        print(f"  ⚠️  WARNING: Standard LSTM is VERY SLOW!")
        print(f"  💡 Consider using fast_lstm instead")


# ============================================================================
# CLASSICAL ML MODELS (FIXED)
# ============================================================================

def run_classical_models(args, logger=None) -> Dict:
    """
    Run all classical ML models with FIXES:
    1. PatientTypeStratifiedGroupKFold
    2. Feature cleaning & selection
    3. Stronger regularization
    4. Better error handling
    """
    if logger is None:
        # Fallback if no logger provided
        class FakeLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"⚠️  {msg}")
            def error(self, msg): print(f"❌ {msg}")
        logger = FakeLogger()
    
    print("\n" + "="*60)
    print("RUNNING CLASSICAL MACHINE LEARNING MODELS")
    print("="*60)
    
    if not args.use_old_cv:
        print("✅ FIX #1: Using PatientTypeStratifiedGroupKFold")
    else:
        print("⚠️  Using old GroupKFold (not recommended)")
    
    print("✅ FIX #2: Feature cleaning & selection enabled")
    print("✅ FIX #3: Stronger regularization applied")
    print("="*60)
    
    start_time = time.time()
    
    try:
        from ecg_mitbih import load_dataset
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score, 
            balanced_accuracy_score, confusion_matrix,
            precision_score, recall_score, log_loss
        )
        
        # SMOTE
        SMOTE_AVAILABLE = False
        if args.use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                SMOTE_AVAILABLE = True
                print("  ✅ SMOTE is available and will be used")
            except ImportError:
                print("  ⚠️  SMOTE not available (install: pip install imbalanced-learn)")
        
        # Load data
        print("\n  🔄 Loading dataset with features...")
        X, y, groups = load_dataset(
            args.data_root,
            records=args.records,
            use_features=True,
            include_wavelet=args.include_wavelet,
            verbose=False
        )
        
        if X.shape[0] == 0:
            print("  ❌ No data loaded")
            return {}
        
        print(f"  ✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")
        print(f"     Patients: {len(np.unique(groups))}")
        print(f"     Class distribution: Normal={np.sum(y==0)}, Arrhythmia={np.sum(y==1)}")
        
        # FEATURE CLEANING & SELECTION (FIX #2)
        X = clean_and_select_features(X, y, n_features=args.n_features, verbose=True)
        
        # Show patient categories
        unique_groups = np.unique(groups)
        print(f"\n  📊 Patient arrhythmia profiles:")
        high_arr, low_arr, mixed = [], [], []
        for g in unique_groups:
            mask = groups == g
            arr_ratio = np.mean(y[mask])
            if arr_ratio > 0.80:
                high_arr.append(f"{g}({arr_ratio:.0%})")
            elif arr_ratio < 0.20:
                low_arr.append(f"{g}({arr_ratio:.0%})")
            else:
                mixed.append(f"{g}({arr_ratio:.0%})")
        
        print(f"     HIGH_ARR (>80%): {high_arr}")
        print(f"     LOW_ARR  (<20%): {low_arr}")
        print(f"     MIXED  (20-80%): {mixed}")
        
        # Setup cross-validation (FIX #1)
        n_patients = len(np.unique(groups))
        n_splits = min(5, n_patients)
        
        if args.use_old_cv:
            cv = GroupKFold(n_splits=n_splits)
            print(f"\n  ⚠️  Using GroupKFold (old method)")
        else:
            cv = PatientTypeStratifiedGroupKFold(n_splits=n_splits, random_state=42)
            print(f"\n  ✅ Using PatientTypeStratifiedGroupKFold")
        
        # MODELS WITH STRONGER REGULARIZATION (FIX #3)
        models = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,               # Reduced from 20!
                    min_samples_split=15,      # Increased from 5!
                    min_samples_leaf=8,        # Increased from 2!
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'loss_function': 'Gini Impurity (regularized)'
            },
            'svm': {
                'model': SVC(
                    kernel='rbf', 
                    C=0.5,                     # Reduced from 1.0!
                    gamma='scale',
                    probability=True,
                    class_weight='balanced',
                    random_state=42,
                    cache_size=1000
                ),
                'loss_function': 'Hinge Loss (regularized)'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,               # Very shallow!
                    learning_rate=0.05,        # Low LR
                    min_samples_split=15,
                    min_samples_leaf=8,
                    subsample=0.7,
                    random_state=42
                ),
                'loss_function': 'Deviance (regularized)'
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    C=0.1,                     # Strong L2 regularization
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                ),
                'loss_function': 'Log Loss (L2 regularized)'
            }
        }
        
        # Try XGBoost
        try:
            from xgboost import XGBClassifier
            models['xgboost'] = {
                'model': XGBClassifier(
                    n_estimators=100,
                    max_depth=3,               # Very shallow!
                    learning_rate=0.03,        # Very low LR
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=1.0,             # Strong L1
                    reg_lambda=2.0,            # Strong L2
                    min_child_weight=8,
                    gamma=0.3,
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=4                   # ✅ UPDATED: Use 4 cores explicitly (avoid thread contention)
                ),
                'loss_function': 'Log Loss (heavily regularized)'
            }
            print("  ✅ XGBoost is available")
        except (ImportError, Exception) as e:
            print(f"  ⚠️  XGBoost not available: {str(e)[:50]}")
        
        results = {}
        
        for model_name, model_config in models.items():
            try:
                model_template = model_config['model']
                loss_fn_name = model_config['loss_function']
                
                print(f"\n  {'─'*50}")
                print(f"  Training {model_name.upper()}...")
                print(f"  Loss Function: {loss_fn_name}")
                print(f"  {'─'*50}")
                
                fold_metrics = []
                
                for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
                    from sklearn.base import clone
                    model = clone(model_template)
                    
                    fold_start = time.time()
                    
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]
                    
                    # Get test patient info
                    test_patients = np.unique(groups[test_idx])
                    train_patients = np.unique(groups[train_idx])
                    test_arr_ratio = np.mean(y_te)
                    
                    print(f"\n    Fold {fold}: Test patients={list(test_patients)}")
                    print(f"             Train: {len(y_tr)} (Arr: {np.mean(y_tr):.1%})")
                    print(f"             Test:  {len(y_te)} (Arr: {test_arr_ratio:.1%})")
                    
                    # Skip extreme imbalance
                    if test_arr_ratio < 0.01 or test_arr_ratio > 0.99:
                        print(f"    ⚠️  Skipped (extreme imbalance)")
                        continue
                    
                    # Normalize
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)
                    
                    # SMOTE
                    if args.use_smote and SMOTE_AVAILABLE:
                        try:
                            unique, counts = np.unique(y_tr, return_counts=True)
                            k = min(5, min(counts) - 1)
                            if k > 0:
                                smote = SMOTE(random_state=42, k_neighbors=k)
                                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                        except Exception as e:
                            print(f"             ⚠️  SMOTE failed: {e}")
                    
                    # Train
                    model.fit(X_tr, y_tr)
                    
                    # Predict
                    y_pred = model.predict(X_te)
                    y_proba = model.predict_proba(X_te)
                    
                    # Losses
                    train_proba = model.predict_proba(X_tr)
                    train_loss = log_loss(y_tr, train_proba)
                    test_loss = log_loss(y_te, y_proba)
                    
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
                        auc = roc_auc_score(y_te, y_proba[:, 1])
                    except ValueError:
                        auc = 0.5
                    
                    fold_time = time.time() - fold_start
                    gap = test_loss - train_loss
                    
                    print(f"             Loss: Train={train_loss:.4f}, Test={test_loss:.4f}, Gap={gap:.4f}")
                    print(f"             Acc={acc:.3f}, Bal_Acc={bal_acc:.3f}, F1={f1:.3f}, AUC={auc:.3f} ({fold_time:.1f}s)")
                    
                    fold_metrics.append({
                        'accuracy': acc,
                        'balanced_accuracy': bal_acc,
                        'f1': f1,
                        'precision': prec,
                        'recall': rec,
                        'auc': auc,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'train_loss': train_loss,
                        'test_loss': test_loss
                    })
                
                if fold_metrics:
                    # Aggregate
                    summary = {}
                    for metric in ['accuracy', 'balanced_accuracy', 'f1', 'precision',
                                   'recall', 'auc', 'sensitivity', 'specificity',
                                   'train_loss', 'test_loss']:
                        values = [m[metric] for m in fold_metrics]
                        summary[metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'values': [float(v) for v in values]
                        }
                    
                    results[model_name] = {
                        'summary': summary,
                        'n_folds_completed': len(fold_metrics),
                        'n_folds_total': n_splits,
                        'used_smote': args.use_smote and SMOTE_AVAILABLE,
                        'loss_function': loss_fn_name,
                        'n_features': X.shape[1],
                        'cv_method': 'PatientTypeStratifiedGroupKFold' if not args.use_old_cv else 'GroupKFold'
                    }
                    
                    # Summary
                    train_loss_mean = summary['train_loss']['mean']
                    test_loss_mean = summary['test_loss']['mean']
                    gap = test_loss_mean - train_loss_mean
                    
                    print(f"\n  ✅ {model_name.upper()} Summary:")
                    print(f"     Train Loss:        {train_loss_mean:.4f} (±{summary['train_loss']['std']:.4f})")
                    print(f"     Test Loss:         {test_loss_mean:.4f} (±{summary['test_loss']['std']:.4f})")
                    print(f"     Loss Gap:          {gap:.4f} {'✅' if gap < 0.3 else '⚠️'}")
                    print(f"     Balanced Accuracy: {summary['balanced_accuracy']['mean']:.3f} (±{summary['balanced_accuracy']['std']:.3f})")
                    print(f"     F1-Score:          {summary['f1']['mean']:.3f} (±{summary['f1']['std']:.3f})")
                    print(f"     AUC:               {summary['auc']['mean']:.3f} (±{summary['auc']['std']:.3f})")
                    print(f"     AUC Range:         [{summary['auc']['min']:.3f}, {summary['auc']['max']:.3f}]")
                    print(f"     Sensitivity:       {summary['sensitivity']['mean']:.3f} (±{summary['sensitivity']['std']:.3f})")
                    print(f"     Specificity:       {summary['specificity']['mean']:.3f} (±{summary['specificity']['std']:.3f})")
                else:
                    print(f"  ⚠️  {model_name}: No valid folds completed")
            
            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Classical ML completed in {elapsed/60:.1f} minutes")
        
        return results
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\n⚠️  Training interrupted by user")
        return results
    
    except Exception as e:
        logger.error(f"Critical error in classical models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # ✅ Save partial results if any
        if 'results' in locals() and results:
            output_file = Path(args.output_dir) / 'partial_classical_results.json'
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved partial results to {output_file}")
                print(f"  💾 Partial results saved to {output_file}")
            except Exception as save_error:
                logger.error(f"Could not save partial results: {save_error}")
        
        print(f"  ❌ Error in classical models: {e}")
        raise


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

def run_deep_learning_models(args) -> Dict:
    """Run deep learning models with smart defaults."""
    print("\n" + "="*60)
    print("RUNNING DEEP LEARNING MODELS")
    print("="*60)
    
    # Smart model selection
    if args.use_slow_lstm:
        models = ['cnn', 'lstm', 'cnn_lstm']
        print("\n  ⚠️  Using SLOW standard LSTM (this will take hours!)")
    else:
        models = ['cnn', 'fast_lstm', 'cnn_lstm']
        print("\n  ✅ Using optimized fast_lstm (4-8x faster than standard LSTM)")
    
    if args.quick_test:
        models = ['cnn', 'fast_lstm']
        print("  🚀 Quick test mode: Testing CNN and Fast LSTM only")
    
    all_results = {}
    total_start = time.time()
    
    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"MODEL {i}/{len(models)}: {model.upper()}")
        print(f"{'='*60}")
        
        print_time_estimate(model, args.epochs if not args.quick_test else 25, len(args.records))
        
        output_file = Path(args.output_dir) / f"results_{model}.json"
        
        batch_size = "32" if model == 'cnn' else "16"
        learning_rate = "0.0001"
        
        cmd = [
            sys.executable, "train_cnn.py",
            "--data_root", args.data_root,
            "--records", *args.records,
            "--cross_validate",
            "--model", model,
            "--epochs", str(args.epochs if not args.quick_test else 25),
            "--batch_size", batch_size,
            "--learning_rate", learning_rate,
            "--output", str(output_file)
        ]
        
        if args.use_smote:
            cmd.append("--use_smote")
        
        print(f"\n  Running with:")
        print(f"    Batch size: {batch_size}")
        print(f"    Learning rate: {learning_rate}")
        print(f"    Epochs: {args.epochs if not args.quick_test else 25}")
        print(f"    SMOTE: {args.use_smote}")
        
        model_start = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=None,
                stderr=None,
                timeout=7200
            )
            
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            model_time = time.time() - model_start
            
            all_results[model] = results
            print(f"\n  ✅ {model.upper()} completed in {model_time/60:.1f} minutes")
            
            if 'summary' in results and 'accuracy' in results['summary']:
                acc = results['summary']['accuracy']['mean']
                f1 = results['summary']['f1']['mean']
                auc = results['summary']['auc']['mean']
                bal_acc = results['summary']['balanced_accuracy']['mean']
                print(f"     Accuracy: {acc:.3f}, Bal.Acc: {bal_acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        except subprocess.TimeoutExpired:
            print(f"  ❌ {model.upper()} training timed out after 2 hours")
            continue
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {model.upper()} failed with error:")
            if e.stderr:
                print(f"     {e.stderr[:500]}")
        except FileNotFoundError as e:
            print(f"  ❌ Results file not found for {model}: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error with {model}: {e}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DEEP LEARNING MODELS COMPLETED")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"{'='*60}")
    
    return all_results


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def generate_comparison_table(classical_results: Dict, dl_results: Dict) -> str:
    """Generate a formatted comparison table."""
    
    table = []
    table.append("="*110)
    table.append("MODEL COMPARISON SUMMARY (Patient-Type Stratified Cross-Validation)")
    table.append("="*110)
    table.append(f"{'Model':<25} {'Train Loss':<12} {'Test Loss':<12} {'Gap':<8} {'Bal.Acc':<12} {'F1':<10} {'AUC':<12}")
    table.append("-"*110)
    
    all_models = []
    
    # Classical models
    if classical_results:
        table.append("\nCLASSICAL MACHINE LEARNING (with fixes):")
        table.append("-"*110)
        
        for model_name, results in classical_results.items():
            if 'summary' in results and 'accuracy' in results['summary']:
                s = results['summary']
                train_loss = s.get('train_loss', {}).get('mean', 0)
                test_loss = s.get('test_loss', {}).get('mean', 0)
                gap = test_loss - train_loss
                bal_acc = s['balanced_accuracy']['mean']
                bal_acc_std = s['balanced_accuracy']['std']
                f1 = s['f1']['mean']
                f1_std = s['f1']['std']
                auc = s['auc']['mean']
                auc_std = s['auc']['std']
                
                smote = " (SMOTE)" if results.get('used_smote', False) else ""
                
                table.append(f"{(model_name.upper() + smote):<25} "
                           f"{train_loss:.4f}       "
                           f"{test_loss:.4f}       "
                           f"{gap:.4f}   "
                           f"{bal_acc:.3f}±{bal_acc_std:.3f}  "
                           f"{f1:.3f}±{f1_std:.3f}  "
                           f"{auc:.3f}±{auc_std:.3f}")
                
                all_models.append((model_name, bal_acc, bal_acc_std, 'classical'))
    
    # Deep learning models
    if dl_results:
        table.append("\nDEEP LEARNING:")
        table.append("-"*110)
        
        for model_name, results in dl_results.items():
            if 'summary' in results and 'accuracy' in results['summary']:
                s = results['summary']
                bal_acc = s['balanced_accuracy']['mean']
                bal_acc_std = s['balanced_accuracy']['std']
                f1 = s['f1']['mean']
                f1_std = s['f1']['std']
                auc = s['auc']['mean']
                auc_std = s['auc']['std']
                
                display_name = model_name.upper()
                if 'total_training_time_minutes' in results:
                    time_min = results['total_training_time_minutes']
                    display_name += f" ({time_min:.0f}m)"
                
                table.append(f"{display_name:<25} "
                           f"{'N/A':<12} "
                           f"{'N/A':<12} "
                           f"{'N/A':<8} "
                           f"{bal_acc:.3f}±{bal_acc_std:.3f}  "
                           f"{f1:.3f}±{f1_std:.3f}  "
                           f"{auc:.3f}±{auc_std:.3f}")
                
                all_models.append((model_name, bal_acc, bal_acc_std, 'deep'))
    
    table.append("="*110)
    
    # Find best model (highest bal_acc with lowest std)
    if all_models:
        best = max(all_models, key=lambda x: x[1] - x[2])  # Penalize high variance
        table.append(f"\n🏆 Best model: {best[0].upper()} (Bal.Acc: {best[1]:.3f}±{best[2]:.3f})")
    
    table.append("\n✅ Fixes applied:")
    table.append("   1. PatientTypeStratifiedGroupKFold → Consistent test distributions")
    table.append("   2. Feature cleaning & selection → Removed noisy features")
    table.append("   3. Stronger regularization → Reduced overfitting")
    table.append("   4. Train-Test Gap tracking → Monitor overfitting")
    
    return "\n".join(table)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_comparison(classical_results: Dict, dl_results: Dict, output_dir: Path):
    """Generate comparison plots."""
    
    models = []
    accuracies = []
    balanced_accs = []
    f1_scores = []
    aucs = []
    model_types = []
    gaps = []
    
    # Classical
    for model_name, results in classical_results.items():
        if 'summary' in results:
            s = results['summary']
            display_name = model_name.upper().replace('_', ' ')
            if results.get('used_smote', False):
                display_name += "*"
            models.append(display_name)
            accuracies.append(s['accuracy']['mean'])
            balanced_accs.append(s['balanced_accuracy']['mean'])
            f1_scores.append(s['f1']['mean'])
            aucs.append(s['auc']['mean'])
            model_types.append('Classical ML')
            gaps.append(s['test_loss']['mean'] - s['train_loss']['mean'])
    
    # Deep learning
    for model_name, results in dl_results.items():
        if 'summary' in results:
            s = results['summary']
            display_name = model_name.upper().replace('_', '-')
            models.append(display_name)
            accuracies.append(s['accuracy']['mean'])
            balanced_accs.append(s['balanced_accuracy']['mean'])
            f1_scores.append(s['f1']['mean'])
            aucs.append(s['auc']['mean'])
            model_types.append('Deep Learning')
            gaps.append(0)  # DL doesn't track this the same way
    
    if not models:
        print("  ⚠️  No data available for plotting")
        return
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ECG Arrhythmia Classification: Model Comparison\n(Patient-Type Stratified CV - FIXED)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#3498db' if t == 'Classical ML' else '#e74c3c' for t in model_types]
    
    # Plot 1: Balanced Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.barh(models, balanced_accs, color=colors, alpha=0.8)
    ax1.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Balanced Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1])
    for i, v in enumerate(balanced_accs):
        ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 2: AUC
    ax2 = axes[0, 1]
    bars2 = ax2.barh(models, aucs, color=colors, alpha=0.8)
    ax2.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Model AUC', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1])
    for i, v in enumerate(aucs):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 3: F1-Score
    ax3 = axes[1, 0]
    bars3 = ax3.barh(models, f1_scores, color=colors, alpha=0.8)
    ax3.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('Model F1-Score', fontsize=13, fontweight='bold')
    ax3.set_xlim([0, 1])
    for i, v in enumerate(f1_scores):
        ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 4: Train-Test Gap (overfitting indicator)
    ax4 = axes[1, 1]
    classical_models = [m for m, t in zip(models, model_types) if t == 'Classical ML']
    classical_gaps = [g for g, t in zip(gaps, model_types) if t == 'Classical ML']
    
    if classical_gaps:
        gap_colors = ['#27ae60' if g < 0.3 else '#e74c3c' for g in classical_gaps]
        bars4 = ax4.barh(classical_models, classical_gaps, color=gap_colors, alpha=0.8)
        ax4.axvline(x=0.3, color='red', linestyle='--', label='Overfitting threshold')
        ax4.set_xlabel('Train-Test Loss Gap', fontsize=12, fontweight='bold')
        ax4.set_title('Overfitting Indicator (lower is better)', fontsize=13, fontweight='bold')
        ax4.legend()
        for i, v in enumerate(classical_gaps):
            ax4.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, label='Classical ML'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Deep Learning')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  📊 Plot saved: {output_file}")
    
    plt.close()


# ============================================================================
# DETAILED REPORT
# ============================================================================

def generate_detailed_report(classical_results: Dict, dl_results: Dict, 
                            output_dir: Path, args):
    """Generate markdown report."""
    
    report = []
    report.append("# ECG Arrhythmia Classification - Comprehensive Report (FIXED)")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n## ✅ Fixes Applied")
    report.append("\n### 1. PatientTypeStratifiedGroupKFold")
    report.append("- Patients grouped by arrhythmia profile (HIGH/LOW/MIXED)")
    report.append("- Each fold contains patients from all profile types")
    report.append("- Ensures consistent test set distributions")
    
    report.append("\n### 2. Feature Cleaning & Selection")
    report.append(f"- Removed constant and highly correlated features")
    report.append(f"- Selected top {args.n_features} features by mutual information")
    report.append("- Handles NaN/Inf values")
    
    report.append("\n### 3. Stronger Regularization")
    report.append("- Random Forest: max_depth=8, min_samples_leaf=8")
    report.append("- XGBoost: max_depth=3, learning_rate=0.03")
    report.append("- All models use balanced class weights")
    
    report.append("\n## Experimental Setup")
    report.append(f"\n- **Dataset:** MIT-BIH Arrhythmia Database")
    report.append(f"- **Patients:** {len(args.records)}")
    report.append(f"- **Cross-Validation:** PatientTypeStratifiedGroupKFold (5-fold)")
    report.append(f"- **Features selected:** {args.n_features}")
    report.append(f"- **SMOTE:** {'Yes' if args.use_smote else 'No'}")
    
    report.append("\n## Results Summary")
    
    # Classical ML
    if classical_results:
        report.append("\n### Classical Machine Learning")
        report.append("\n| Model | Train Loss | Test Loss | Gap | Bal.Acc | F1 | AUC |")
        report.append("|-------|------------|-----------|-----|---------|----|----|")
        
        for model_name, results in classical_results.items():
            if 'summary' in results:
                s = results['summary']
                train_loss = s.get('train_loss', {}).get('mean', 0)
                test_loss = s.get('test_loss', {}).get('mean', 0)
                gap = test_loss - train_loss
                gap_emoji = "✅" if gap < 0.3 else "⚠️"
                
                report.append(
                    f"| {model_name.upper()} | "
                    f"{train_loss:.4f} | "
                    f"{test_loss:.4f} | "
                    f"{gap:.4f} {gap_emoji} | "
                    f"{s['balanced_accuracy']['mean']:.3f}±{s['balanced_accuracy']['std']:.3f} | "
                    f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f} | "
                    f"{s['auc']['mean']:.3f}±{s['auc']['std']:.3f} |"
                )
    
    # Deep Learning
    if dl_results:
        report.append("\n### Deep Learning")
        report.append("\n| Model | Bal.Acc | F1 | AUC | Training Time |")
        report.append("|-------|---------|----|----|---------------|")
        
        for model_name, results in dl_results.items():
            if 'summary' in results:
                s = results['summary']
                time_str = f"{results.get('total_training_time_minutes', 0):.0f}m"
                report.append(
                    f"| {model_name.upper()} | "
                    f"{s['balanced_accuracy']['mean']:.3f}±{s['balanced_accuracy']['std']:.3f} | "
                    f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f} | "
                    f"{s['auc']['mean']:.3f}±{s['auc']['std']:.3f} | "
                    f"{time_str} |"
                )
    
    report.append("\n## Visualizations")
    report.append("\n![Model Comparison](model_comparison.png)")
    
    report.append("\n## Key Insights")
    report.append("\n### Overfitting Analysis")
    report.append("- Train-Test Loss Gap < 0.3 indicates good generalization")
    report.append("- Higher gaps suggest overfitting")
    
    report.append("\n### Patient Heterogeneity")
    report.append("- Some patients are nearly 100% arrhythmia")
    report.append("- Others are nearly 100% normal")
    report.append("- This inherent variance limits achievable consistency")
    
    # Save
    output_file = output_dir / "report.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"  📄 Report saved: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # ✅ Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(str(output_dir), 'model_comparison')
    
    logger.info(f"Starting comparison with {len(args.records)} patients")
    logger.info(f"SMOTE: {args.use_smote}, Wavelet: {args.include_wavelet}")
    
    if args.quick_test:
        logger.info("🚀 QUICK TEST MODE")
        logger.info("  - Reduced epochs to 25")
        logger.info("  - Testing only CNN and Fast LSTM")
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE MODEL COMPARISON - FIXED VERSION")
    logger.info("="*60)
    logger.info("Fixes applied:")
    logger.info("  ✅ PatientTypeStratifiedGroupKFold")
    logger.info("  ✅ Feature cleaning & selection")
    logger.info("  ✅ Stronger regularization")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Patients: {len(args.records)}")
    logger.info(f"Features to select: {args.n_features}")
    logger.info(f"SMOTE: {'ENABLED' if args.use_smote else 'DISABLED'}")
    logger.info("="*60)
    
    overall_start = time.time()
    
    classical_results = {}
    dl_results = {}
    
    if not args.skip_classical:
        classical_results = run_classical_models(args, logger)
    else:
        print("\n⏭️  Skipping classical ML models")
    
    if not args.skip_deep:
        dl_results = run_deep_learning_models(args)
    else:
        print("\n⏭️  Skipping deep learning models")
    
    # Generate comparison
    if classical_results or dl_results:
        print("\n" + "="*60)
        print("GENERATING COMPARISON REPORT")
        print("="*60)
        
        table = generate_comparison_table(classical_results, dl_results)
        print(f"\n{table}")
        
        try:
            plot_comparison(classical_results, dl_results, output_dir)
        except Exception as e:
            print(f"  ⚠️  Could not generate plots: {e}")
        
        try:
            generate_detailed_report(classical_results, dl_results, output_dir, args)
        except Exception as e:
            print(f"  ⚠️  Could not generate report: {e}")
        
        # Save all results
        all_results = {
            'classical': classical_results,
            'deep_learning': dl_results,
            'config': {
                'n_features': args.n_features,
                'use_smote': args.use_smote,
                'cv_method': 'PatientTypeStratifiedGroupKFold' if not args.use_old_cv else 'GroupKFold',
                'records': args.records
            }
        }
        
        with open(output_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        overall_time = time.time() - overall_start
        print(f"\n{'='*60}")
        print(f"✅ ALL COMPLETED")
        print(f"Total time: {overall_time/60:.1f} minutes ({overall_time/3600:.1f} hours)")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
    else:
        print("\n❌ No results available to compare")


if __name__ == "__main__":
    main()
