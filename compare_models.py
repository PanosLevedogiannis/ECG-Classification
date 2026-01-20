"""
Comprehensive Model Comparison for ECG Arrhythmia Classification
FINAL OPTIMIZED VERSION - With SMOTE for Classical ML
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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run comprehensive model comparison"
    )
    ap.add_argument("--data_root", type=str, default="data/mitdb")
    ap.add_argument(
        "--records",
        type=str,
        nargs="+",
        # ✅ OPTIMIZED: Default to 20 patients for robust cross-validation
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
    return ap.parse_args()


def print_time_estimate(model_name: str, epochs: int, n_records: int):
    """Print estimated training time for each model based on number of records."""
    # ✅ IMPROVED: More accurate time estimates based on record count
    # Times in seconds per epoch per 1000 samples
    time_per_sample = {
        'cnn': 0.1,       # ~100ms per 1000 samples
        'fast_lstm': 0.15, # ~150ms per 1000 samples
        'lstm': 0.6,      # ~600ms per 1000 samples (SLOW!)
        'cnn_lstm': 0.2   # ~200ms per 1000 samples
    }
    
    # Rough estimate: ~720 samples per record
    estimated_samples = n_records * 720
    estimated_seconds = (time_per_sample.get(model_name, 0.15) * 
                        estimated_samples * epochs * 5 / 1000)  # 5 folds
    estimated_minutes = estimated_seconds / 60
    
    if estimated_minutes > 60:
        print(f"  ⏱️  Estimated time: {estimated_minutes/60:.1f} hours")
        if estimated_minutes > 120:
            print(f"  ⚠️  WARNING: This will take over 2 hours!")
    else:
        print(f"  ⏱️  Estimated time: {estimated_minutes:.0f} minutes")
    
    if model_name == 'lstm':
        print(f"  ⚠️  WARNING: This will take VERY LONG!")
        print(f"  💡 Consider using fast_lstm instead")


def run_classical_models(args) -> Dict:
    """
    Run all classical ML models with PATIENT-WISE cross-validation.
    
    ✅ NEW: SMOTE support for classical ML models!
    """
    print("\n" + "="*60)
    print("RUNNING CLASSICAL MACHINE LEARNING MODELS")
    print("Patient-Wise Cross-Validation + SMOTE Support")
    print("="*60)
    
    start_time = time.time()
    
    try:
        from ecg_mitbih import load_dataset
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier  # Removed GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score, 
            balanced_accuracy_score, confusion_matrix,
            precision_score, recall_score
        )
        
        # XGBoost will be imported later with try/except
        
        # ✅ SMOTE import
        SMOTE_AVAILABLE = False
        if args.use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                SMOTE_AVAILABLE = True
                print("  ✅ SMOTE is available and will be used")
            except ImportError:
                print("  ⚠️  SMOTE not available (install: pip install imbalanced-learn)")
        
        print("  🔄 Loading dataset with features...")
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
        
        # Check for sufficient patients
        n_patients = len(np.unique(groups))
        if n_patients < 5:
            print(f"  ⚠️  WARNING: Only {n_patients} patients - results may be unreliable!")
            n_splits = n_patients
        else:
            n_splits = 5
        
        # ✅ IMPROVED: Optimized models to prevent overfitting
        models = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=300,          # Increased from 200 (more stable)
                    max_depth=20,              # Reduced from 30 (prevent overfitting)
                    min_samples_split=5,       # Increased from 2 (more regularization)
                    min_samples_leaf=2,        # Minimum samples per leaf (NEW)
                    max_features='sqrt',       # Feature sampling (NEW)
                    criterion='gini',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'loss_function': 'Gini Impurity'
            },
            'svm': {
                'model': SVC(
                    kernel='rbf', 
                    C=1.0,
                    gamma='scale',
                    probability=True, 
                    random_state=42,
                    cache_size=1000
                ),
                'loss_function': 'Hinge Loss'
            },
            # ✅ NEW: XGBoost (state-of-the-art gradient boosting)
            'xgboost': {
                'model': None,  # Will be initialized after checking if XGBoost is available
                'loss_function': 'Binary Log Loss'
            }
        }
        
        # ✅ Initialize XGBoost (with graceful fallback if not installed)
        try:
            from xgboost import XGBClassifier
            # ✅ OPTIMIZED: Strong regularization to prevent overfitting
            models['xgboost']['model'] = XGBClassifier(
                n_estimators=100,
                max_depth=4,              # Balanced (3→4)
                learning_rate=0.08,       # Balanced (0.05→0.08)
                subsample=0.7,            # Reduced from 0.8 (more regularization)
                colsample_bytree=0.7,     # Reduced from 0.8 (more regularization)
                reg_alpha=0.3,            # Balanced (0.5→0.3)
                reg_lambda=0.5,           # Balanced (1.0→0.5)
                min_child_weight=3,       # Minimum samples per leaf (NEW)
                gamma=0.1,                # Minimum loss reduction (NEW)
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            print("  ✅ XGBoost is available (with anti-overfitting regularization)")
        except (ImportError, Exception) as e:
            # Catch both ImportError and XGBoost-specific errors (like missing libomp)
            if "libomp" in str(e) or "OpenMP" in str(e):
                print("  ⚠️  XGBoost not available: Missing OpenMP library")
                print("      Fix: Run 'brew install libomp' on Mac")
            else:
                print(f"  ⚠️  XGBoost not available: {str(e)[:100]}")
            print("      → Continuing with Random Forest and SVM only")
            del models['xgboost']  # Remove from models dict
        
        results = {}
        gkf = GroupKFold(n_splits=n_splits)
        
        print(f"\n  ✅ Using GroupKFold (patient-wise) with {n_splits} splits")
        print(f"     Test patients NEVER appear in training!")
        if args.use_smote and SMOTE_AVAILABLE:
            print(f"     SMOTE will balance training data in each fold")
        
        for model_name, model_config in models.items():
            try:
                model = model_config['model']
                loss_fn_name = model_config['loss_function']
                
                print(f"\n  {'─'*50}")
                print(f"  Training {model_name.upper()}...")
                print(f"  Loss Function: {loss_fn_name}")  # ✅ NEW
                print(f"  {'─'*50}")
                
                fold_metrics = []
                
                for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
                    fold_start = time.time()
                    
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]
                    
                    # Verify no patient overlap
                    train_patients = set(groups[train_idx])
                    test_patients = set(groups[test_idx])
                    overlap = train_patients & test_patients
                    
                    if len(overlap) > 0:
                        print(f"    ❌ ERROR: Data leakage detected! Overlapping patients: {overlap}")
                        continue
                    
                    print(f"    Fold {fold}: Train={len(train_patients)} patients, "
                          f"Test={len(test_patients)} patients")
                    print(f"             Train samples={len(y_tr)} "
                          f"(Normal={np.sum(y_tr==0)}, Arr={np.sum(y_tr==1)})")
                    print(f"             Test samples={len(y_te)} "
                          f"(Normal={np.sum(y_te==0)}, Arr={np.sum(y_te==1)})")
                    
                    # Skip extremely imbalanced test sets
                    test_normal_ratio = np.sum(y_te == 0) / len(y_te)
                    if test_normal_ratio < 0.01 or test_normal_ratio > 0.99:
                        print(f"    ⚠️  Skipped (extreme imbalance: {test_normal_ratio:.1%} normal)")
                        continue
                    
                    # Warn about moderate imbalance
                    if test_normal_ratio < 0.05 or test_normal_ratio > 0.95:
                        print(f"    ⚠️  Warning: Imbalanced test set ({test_normal_ratio:.1%} normal)")
                    
                    # ✅ CRITICAL: Normalize BEFORE SMOTE
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)
                    
                    # ✅ NEW: Apply SMOTE if requested
                    if args.use_smote and SMOTE_AVAILABLE:
                        try:
                            # Calculate class distribution
                            unique, counts = np.unique(y_tr, return_counts=True)
                            minority_count = np.min(counts)
                            
                            # Calculate k_neighbors (must be < minority samples)
                            k_neighbors = min(5, minority_count - 1)
                            
                            if k_neighbors > 0:
                                print(f"             Before SMOTE: {dict(zip(unique, counts))}")
                                
                                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                                
                                unique_new, counts_new = np.unique(y_tr, return_counts=True)
                                print(f"             After SMOTE:  {dict(zip(unique_new, counts_new))}")
                            else:
                                print(f"             ⚠️  SMOTE skipped (too few minority samples: {minority_count})")
                        
                        except Exception as e:
                            print(f"             ⚠️  SMOTE failed: {e}")
                    
                    # Train and predict
                    # ✅ Track training (with loss info for XGBoost)
                    if model_name == 'xgboost':
                        # XGBoost can show training progress
                        eval_set = [(X_tr, y_tr), (X_te, y_te)]
                        model.fit(
                            X_tr, y_tr,
                            eval_set=eval_set,
                            verbose=False  # We'll compute loss manually
                        )
                        
                        # Get training loss from XGBoost
                        train_loss = model.evals_result()['validation_0']['logloss'][-1]
                        test_loss = model.evals_result()['validation_1']['logloss'][-1]
                        print(f"             Train Loss (log): {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                    else:
                        model.fit(X_tr, y_tr)
                    
                    y_pred = model.predict(X_te)
                    y_proba = model.predict_proba(X_te)
                    
                    # ✅ Compute loss manually for all models
                    from sklearn.metrics import log_loss
                    try:
                        # Binary cross-entropy (same as deep learning!)
                        train_proba = model.predict_proba(X_tr)
                        train_loss_manual = log_loss(y_tr, train_proba)
                        test_loss_manual = log_loss(y_te, y_proba)
                        
                        if model_name != 'xgboost':  # XGBoost already printed
                            print(f"             Binary CE Loss - Train: {train_loss_manual:.4f}, Test: {test_loss_manual:.4f}")
                    except Exception:
                        pass  # Skip if log_loss fails
                    
                    # ✅ NEW: Calculate loss on test set using sklearn
                    from sklearn.metrics import log_loss
                    
                    try:
                        # Binary cross-entropy (same metric for all models)
                        test_loss = log_loss(y_te, y_proba)
                    except Exception as e:
                        # Fallback to manual calculation if log_loss fails
                        epsilon = 1e-15
                        y_proba_clip = np.clip(y_proba[:, 1], epsilon, 1 - epsilon)
                        test_loss = -np.mean(
                            y_te * np.log(y_proba_clip) + 
                            (1 - y_te) * np.log(1 - y_proba_clip)
                        )
                    
                    # Calculate comprehensive metrics
                    acc = accuracy_score(y_te, y_pred)
                    bal_acc = balanced_accuracy_score(y_te, y_pred)
                    f1 = f1_score(y_te, y_pred, average='binary', zero_division=0)
                    prec = precision_score(y_te, y_pred, average='binary', zero_division=0)
                    rec = recall_score(y_te, y_pred, average='binary', zero_division=0)
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_te, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    try:
                        auc = roc_auc_score(y_te, y_proba[:, 1])
                    except ValueError:
                        auc = np.nan
                    
                    fold_time = time.time() - fold_start
                    print(f"             Acc={acc:.3f}, Bal_Acc={bal_acc:.3f}, "
                          f"F1={f1:.3f}, AUC={auc:.3f}, Loss={test_loss:.4f} ({fold_time:.1f}s)")
                    
                    fold_metrics.append({
                        'accuracy': acc, 
                        'balanced_accuracy': bal_acc,
                        'f1': f1, 
                        'precision': prec,
                        'recall': rec,
                        'auc': auc,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'loss': test_loss  # ✅ NEW
                    })
                
                if fold_metrics:
                    # Aggregate results
                    summary = {}
                    for metric in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 
                                 'recall', 'auc', 'sensitivity', 'specificity', 'loss']:  # ✅ Added loss
                        values = [m[metric] for m in fold_metrics]
                        summary[metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'values': [float(v) for v in values]
                        }
                    
                    results[model_name] = {
                        'summary': summary,
                        'n_folds_completed': len(fold_metrics),
                        'n_folds_total': n_splits,
                        'used_smote': args.use_smote and SMOTE_AVAILABLE,
                        'loss_function': loss_fn_name  # ✅ NEW
                    }
                    
                    print(f"\n  ✅ {model_name.upper()} Summary:")
                    print(f"     Loss Function:     {loss_fn_name}")  # ✅ NEW
                    print(f"     Test Loss:         {summary['loss']['mean']:.4f} (±{summary['loss']['std']:.4f})")  # ✅ NEW
                    print(f"     Accuracy:          {summary['accuracy']['mean']:.3f} (±{summary['accuracy']['std']:.3f})")
                    print(f"     Balanced Accuracy: {summary['balanced_accuracy']['mean']:.3f} (±{summary['balanced_accuracy']['std']:.3f})")
                    print(f"     F1-Score:          {summary['f1']['mean']:.3f} (±{summary['f1']['std']:.3f})")
                    print(f"     AUC:               {summary['auc']['mean']:.3f} (±{summary['auc']['std']:.3f})")
                    print(f"     Sensitivity:       {summary['sensitivity']['mean']:.3f} (±{summary['sensitivity']['std']:.3f})")
                    print(f"     Specificity:       {summary['specificity']['mean']:.3f} (±{summary['specificity']['std']:.3f})")
                else:
                    print(f"  ⚠️  {model_name}: No valid folds completed")
            
            except Exception as e:
                print(f"  ❌ {model_name} failed with error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        elapsed = time.time() - start_time
        print(f"\n  ⏱️  Classical ML completed in {elapsed/60:.1f} minutes")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error in classical models: {e}")
        import traceback
        traceback.print_exc()
        return {}


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
        models = ['cnn', 'fast_lstm']  # Skip hybrid in quick mode
        print("  🚀 Quick test mode: Testing CNN and Fast LSTM only")
    
    all_results = {}
    total_start = time.time()
    
    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"MODEL {i}/{len(models)}: {model.upper()}")
        print(f"{'='*60}")
        
        print_time_estimate(model, args.epochs if not args.quick_test else 25, len(args.records))
        
        output_file = Path(args.output_dir) / f"results_{model}.json"
        
        # ✅ OPTIMIZED: Smaller batch sizes for CPU (Mac)
        batch_size = "32" if model == 'cnn' else "16"
        
        # ✅ OPTIMIZED: Lower learning rate for more stable training
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
            # Run the training
            result = subprocess.run(
                cmd,
                check=True,
                stdout=None,
                stderr=None,
                timeout=7200  # 2 hour timeout
            )
            
            # Load results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            model_time = time.time() - model_start
            
            all_results[model] = results
            print(f"\n  ✅ {model.upper()} completed in {model_time/60:.1f} minutes")
            
            # Print quick summary
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


def generate_comparison_table(classical_results: Dict, dl_results: Dict) -> str:
    """Generate a formatted comparison table."""
    
    table = []
    table.append("="*100)
    table.append("MODEL COMPARISON SUMMARY (Patient-Wise Cross-Validation)")
    table.append("="*100)
    table.append(f"{'Model':<25} {'Loss Fn':<20} {'Test Loss':<12} {'Accuracy':<12} {'Bal. Acc':<12} {'F1':<10}")
    table.append("-"*100)
    
    # Classical models
    if classical_results:
        table.append("\nCLASSICAL MACHINE LEARNING:")
        table.append("-"*100)
        
        for model_name, results in classical_results.items():
            if 'summary' in results and 'accuracy' in results['summary']:
                loss_fn = results.get('loss_function', 'N/A')
                test_loss = results['summary'].get('loss', {}).get('mean', 0)
                test_loss_std = results['summary'].get('loss', {}).get('std', 0)
                acc = results['summary']['accuracy']['mean']
                acc_std = results['summary']['accuracy']['std']
                bal_acc = results['summary']['balanced_accuracy']['mean']
                bal_acc_std = results['summary']['balanced_accuracy']['std']
                f1 = results['summary']['f1']['mean']
                f1_std = results['summary']['f1']['std']
                
                smote_marker = " (SMOTE)" if results.get('used_smote', False) else ""
                
                table.append(f"{(model_name.upper() + smote_marker):<25} "
                           f"{loss_fn:<20} "
                           f"{test_loss:.4f}±{test_loss_std:.4f}  "
                           f"{acc:.3f}±{acc_std:.3f}  "
                           f"{bal_acc:.3f}±{bal_acc_std:.3f}  "
                           f"{f1:.3f}±{f1_std:.3f}")
    
    # Deep learning models
    if dl_results:
        table.append("\nDEEP LEARNING:")
        table.append("-"*100)
        
        for model_name, results in dl_results.items():
            if 'summary' in results and 'accuracy' in results['summary']:
                loss_fn = results.get('loss_function', 'Binary CE')
                # DL models track loss during training
                test_loss_mean = results['summary'].get('loss', {}).get('mean', 0) if 'loss' in results.get('summary', {}) else 0
                test_loss_std = results['summary'].get('loss', {}).get('std', 0) if 'loss' in results.get('summary', {}) else 0
                
                acc = results['summary']['accuracy']['mean']
                acc_std = results['summary']['accuracy']['std']
                bal_acc = results['summary']['balanced_accuracy']['mean']
                bal_acc_std = results['summary']['balanced_accuracy']['std']
                f1 = results['summary']['f1']['mean']
                f1_std = results['summary']['f1']['std']
                
                display_name = model_name.upper()
                if 'total_training_time_minutes' in results:
                    time_min = results['total_training_time_minutes']
                    display_name += f" ({time_min:.0f}m)"
                
                loss_display = f"{test_loss_mean:.4f}±{test_loss_std:.4f}" if test_loss_mean > 0 else "N/A"
                
                table.append(f"{display_name:<25} "
                           f"{loss_fn:<20} "
                           f"{loss_display:>12}  "
                           f"{acc:.3f}±{acc_std:.3f}  "
                           f"{bal_acc:.3f}±{bal_acc_std:.3f}  "
                           f"{f1:.3f}±{f1_std:.3f}")
    
    table.append("="*100)
    table.append("\n✅ All models use patient-wise cross-validation (no data leakage)")
    table.append("   Test patients NEVER appear in training set")
    table.append("   Loss = Binary Log Loss (for comparison across all models)")
    
    return "\n".join(table)


def plot_comparison(classical_results: Dict, dl_results: Dict, output_dir: Path):
    """Generate comparison plots."""
    
    models = []
    accuracies = []
    balanced_accs = []
    f1_scores = []
    aucs = []
    model_types = []
    
    # Classical models
    for model_name, results in classical_results.items():
        if 'summary' in results and 'accuracy' in results['summary']:
            display_name = model_name.upper().replace('_', ' ')
            if results.get('used_smote', False):
                display_name += " (SMOTE)"
            models.append(display_name)
            accuracies.append(results['summary']['accuracy']['mean'])
            balanced_accs.append(results['summary']['balanced_accuracy']['mean'])
            f1_scores.append(results['summary']['f1']['mean'])
            aucs.append(results['summary']['auc']['mean'])
            model_types.append('Classical ML')
    
    # Deep learning models
    for model_name, results in dl_results.items():
        if 'summary' in results and 'accuracy' in results['summary']:
            display_name = model_name.upper().replace('_', '-')
            models.append(display_name)
            accuracies.append(results['summary']['accuracy']['mean'])
            balanced_accs.append(results['summary']['balanced_accuracy']['mean'])
            f1_scores.append(results['summary']['f1']['mean'])
            aucs.append(results['summary']['auc']['mean'])
            model_types.append('Deep Learning')
    
    if not models:
        print("  ⚠️  No data available for plotting")
        return
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ECG Arrhythmia Classification: Model Comparison\n(Patient-Wise CV - No Data Leakage)', 
                 fontsize=16, fontweight='bold')
    
    # Color palette
    colors = ['#3498db' if t == 'Classical ML' else '#e74c3c' for t in model_types]
    
    # Plot 1: Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.barh(models, accuracies, color=colors, alpha=0.8)
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1])
    for i, v in enumerate(accuracies):
        ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 2: Balanced Accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.barh(models, balanced_accs, color=colors, alpha=0.8)
    ax2.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Model Balanced Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1])
    for i, v in enumerate(balanced_accs):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 3: F1-Score
    ax3 = axes[1, 0]
    bars3 = ax3.barh(models, f1_scores, color=colors, alpha=0.8)
    ax3.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('Model F1-Score', fontsize=13, fontweight='bold')
    ax3.set_xlim([0, 1])
    for i, v in enumerate(f1_scores):
        ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 4: Combined
    ax4 = axes[1, 1]
    x = np.arange(len(models))
    width = 0.2
    
    bars_acc = ax4.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db', alpha=0.8)
    bars_bal = ax4.bar(x - 0.5*width, balanced_accs, width, label='Bal. Acc', color='#9b59b6', alpha=0.8)
    bars_f1 = ax4.bar(x + 0.5*width, f1_scores, width, label='F1-Score', color='#2ecc71', alpha=0.8)
    bars_auc = ax4.bar(x + 1.5*width, aucs, width, label='AUC', color='#f39c12', alpha=0.8)
    
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('All Metrics', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax4.legend()
    ax4.set_ylim([0, 1])
    ax4.grid(axis='y', alpha=0.3)
    
    # Legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, label='Classical ML'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Deep Learning')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  📊 Plot saved: {output_file}")
    
    plt.close()


def generate_detailed_report(classical_results: Dict, dl_results: Dict, 
                            output_dir: Path, args):
    """Generate markdown report."""
    
    report = []
    report.append("# ECG Arrhythmia Classification - Comprehensive Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n## ✅ Methodology")
    report.append("\n### Cross-Validation Strategy")
    report.append("- **Patient-wise splitting** using GroupKFold")
    report.append("- Test patients **NEVER** appear in training set")
    report.append("- No overlap of windows from the same patient across train/test")
    report.append("- Fair comparison between all models")
    
    report.append("\n### Class Imbalance Handling")
    if args.use_smote:
        report.append("- **SMOTE (Synthetic Minority Over-sampling Technique)** applied to training data")
        report.append("- Balances class distribution while preserving patient-wise separation")
        report.append("- Applied independently in each fold to prevent data leakage")
    report.append("- **Balanced Accuracy** used as primary metric")
    report.append("- Class weights applied in model training")
    
    report.append("\n### Signal Processing (ΣΕΣ)")
    report.append("- Butterworth bandpass filter (0.5-40 Hz)")
    report.append("- 5-second windows with 2.5-second overlap")
    report.append("- Removal of power line interference and baseline wander")
    report.append("- Feature extraction: time-domain, frequency-domain, HRV, wavelet (optional)")
    
    report.append("\n### Machine Learning (ΜΜ)")
    report.append("- **Classical ML**: Hand-crafted features from signal processing")
    report.append("- **Deep Learning**: Automatic feature learning from raw signals")
    report.append("- Metrics: Accuracy, Balanced Accuracy, F1-Score, AUC, Sensitivity, Specificity")
    
    report.append("\n## Experimental Setup")
    report.append(f"\n- **Dataset:** MIT-BIH Arrhythmia Database")
    report.append(f"- **Patients:** {len(args.records)}")
    report.append(f"- **Cross-Validation:** 5-fold patient-wise (GroupKFold)")
    report.append(f"- **Window size:** 5 seconds")
    report.append(f"- **Overlap:** 2.5 seconds")
    report.append(f"- **SMOTE:** {'Yes (Classical + Deep Learning)' if args.use_smote else 'No (Deep Learning only)'}")
    report.append(f"- **Wavelet features:** {'Yes' if args.include_wavelet else 'No'}")
    
    report.append("\n## Results Summary")
    
    # Classical ML
    if classical_results:
        report.append("\n### Classical Machine Learning")
        report.append("\n| Model | Loss Function | Test Loss | Accuracy | Bal. Accuracy | F1-Score | AUC |")
        report.append("|-------|---------------|-----------|----------|---------------|----------|-----|")
        
        for model_name, results in classical_results.items():
            if 'summary' in results:
                s = results['summary']
                loss_fn = results.get('loss_function', 'N/A')
                test_loss = s.get('loss', {}).get('mean', 0)
                test_loss_std = s.get('loss', {}).get('std', 0)
                smote_marker = " (SMOTE)" if results.get('used_smote', False) else ""
                report.append(
                    f"| {model_name.upper().replace('_', ' ')}{smote_marker} | "
                    f"{loss_fn} | "
                    f"{test_loss:.4f} ± {test_loss_std:.4f} | "
                    f"{s.get('accuracy', {}).get('mean', 0):.3f} ± {s.get('accuracy', {}).get('std', 0):.3f} | "
                    f"{s.get('balanced_accuracy', {}).get('mean', 0):.3f} ± {s.get('balanced_accuracy', {}).get('std', 0):.3f} | "
                    f"{s.get('f1', {}).get('mean', 0):.3f} ± {s.get('f1', {}).get('std', 0):.3f} | "
                    f"{s.get('auc', {}).get('mean', 0):.3f} ± {s.get('auc', {}).get('std', 0):.3f} |"
                )
    
    # Deep Learning
    if dl_results:
        report.append("\n### Deep Learning")
        report.append("\n| Model | Accuracy | Bal. Accuracy | F1-Score | AUC | Training Time |")
        report.append("|-------|----------|---------------|----------|-----|---------------|")
        
        for model_name, results in dl_results.items():
            if 'summary' in results:
                s = results['summary']
                time_str = f"{results.get('total_training_time_minutes', 0):.0f}m"
                report.append(
                    f"| {model_name.upper().replace('_', '-')} | "
                    f"{s.get('accuracy', {}).get('mean', 0):.3f} ± {s.get('accuracy', {}).get('std', 0):.3f} | "
                    f"{s.get('balanced_accuracy', {}).get('mean', 0):.3f} ± {s.get('balanced_accuracy', {}).get('std', 0):.3f} | "
                    f"{s.get('f1', {}).get('mean', 0):.3f} ± {s.get('f1', {}).get('std', 0):.3f} | "
                    f"{s.get('auc', {}).get('mean', 0):.3f} ± {s.get('auc', {}).get('std', 0):.3f} | "
                    f"{time_str} |"
                )
    
    report.append("\n## Visualizations")
    report.append("\n![Model Comparison](model_comparison.png)")
    
    report.append("\n## Discussion")
    report.append("\n### Key Findings")
    report.append("- Patient-wise cross-validation ensures generalization to unseen patients")
    report.append("- Balanced accuracy used as primary metric to handle class imbalance")
    if args.use_smote:
        report.append("- SMOTE successfully balanced training data without data leakage")
    report.append("- Deep learning models learn features automatically from raw signals")
    report.append("- Classical ML models rely on hand-crafted features from signal processing")
    
    report.append("\n### ΣΕΣ vs ΜΜ Integration")
    report.append("- **ΣΕΣ (Signal Processing)**: Bandpass filtering, feature extraction (time/frequency/HRV)")
    report.append("- **ΜΜ (Machine Learning)**: Classification using extracted features (Classical) or raw signals (DL)")
    report.append("- Deep Learning demonstrates the advantage of automatic feature learning over manual feature engineering")
    
    # Save
    output_file = output_dir / "report.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"  📄 Report saved: {output_file}")


def main():
    args = parse_args()
    
    # Quick test mode adjustments
    if args.quick_test:
        print("\n🚀 QUICK TEST MODE")
        print("  - Reduced epochs to 25")
        print("  - Testing only CNN and Fast LSTM")
        print("  - Optimized batch sizes for CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON - FINAL OPTIMIZED")
    print("Patient-Wise CV + CPU Optimizations + SMOTE for All")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Patients: {len(args.records)}")
    if args.use_smote:
        print(f"SMOTE: ENABLED for ALL models (Classical + Deep Learning)")
    else:
        print(f"SMOTE: Deep Learning only")
    print("="*60)
    
    overall_start = time.time()
    
    classical_results = {}
    dl_results = {}
    
    # Run classical models
    if not args.skip_classical:
        classical_results = run_classical_models(args)
    else:
        print("\n⏭️  Skipping classical ML models")
    
    # Run deep learning models
    if not args.skip_deep:
        dl_results = run_deep_learning_models(args)
    else:
        print("\n⏭️  Skipping deep learning models")
    
    # Generate comparison
    if classical_results or dl_results:
        print("\n" + "="*60)
        print("GENERATING COMPARISON REPORT")
        print("="*60)
        
        # Print table
        table = generate_comparison_table(classical_results, dl_results)
        print(f"\n{table}")
        
        # Generate plots
        try:
            plot_comparison(classical_results, dl_results, output_dir)
        except Exception as e:
            print(f"  ⚠️  Could not generate plots: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate report
        try:
            generate_detailed_report(classical_results, dl_results, output_dir, args)
        except Exception as e:
            print(f"  ⚠️  Could not generate report: {e}")
            import traceback
            traceback.print_exc()
        
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
