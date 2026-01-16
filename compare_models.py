"""
Comprehensive Model Comparison for ECG Arrhythmia Classification
OPTIMIZED VERSION - Uses fast_lstm by default
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
    default=[
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234"
    ],
)
    
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--use_smote", action="store_true",
                   help="Use SMOTE for all models")
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


def print_time_estimate(model_name: str, epochs: int):
    """Print estimated training time for each model."""
    # Rough estimates based on typical hardware
    time_per_epoch = {
        'cnn': 10,  # seconds
        'fast_lstm': 15,  # seconds
        'lstm': 60,  # seconds (SLOW!)
        'cnn_lstm': 20  # seconds
    }
    
    estimated_seconds = time_per_epoch.get(model_name, 20) * epochs * 5  # 5 folds
    estimated_minutes = estimated_seconds / 60
    
    if estimated_minutes > 60:
        print(f"  ⏱️  Estimated time: {estimated_minutes/60:.1f} hours")
    else:
        print(f"  ⏱️  Estimated time: {estimated_minutes:.0f} minutes")
    
    if model_name == 'lstm':
        print(f"  ⚠️  WARNING: This will take VERY LONG!")
        print(f"  💡 Consider using fast_lstm instead")


def run_classical_models(args) -> Dict:
    """Run all classical ML models."""
    print("\n" + "="*60)
    print("RUNNING CLASSICAL MACHINE LEARNING MODELS")
    print("="*60)
    
    start_time = time.time()
    
    try:
        from ecg_mitbih import load_dataset
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
        
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
        
        # Define classical models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=30,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            ),
            'svm': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42,
                cache_size=1000  # Increase cache for speed
            ),
        }
        
        results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            try:
                print(f"\n  {'─'*50}")
                print(f"  Training {model_name.upper()}...")
                print(f"  {'─'*50}")
                
                fold_metrics = []
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
                    fold_start = time.time()
                    
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]
                    
                    # Skip extremely imbalanced test sets
                    test_normal_ratio = np.sum(y_te == 0) / len(y_te)
                    if test_normal_ratio < 0.05 or test_normal_ratio > 0.95:
                        print(f"    Fold {fold}: Skipped (imbalanced)")
                        continue
                    
                    # Normalize
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)
                    
                    # Train and predict
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    y_proba = model.predict_proba(X_te)
                    
                    # Calculate metrics
                    acc = accuracy_score(y_te, y_pred)
                    bal_acc = balanced_accuracy_score(y_te, y_pred)
                    f1 = f1_score(y_te, y_pred, average='binary', zero_division=0)
                    
                    try:
                        auc = roc_auc_score(y_te, y_proba[:, 1])
                    except ValueError:
                        auc = np.nan
                    
                    fold_time = time.time() - fold_start
                    print(f"    Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f} ({fold_time:.1f}s)")
                    
                    fold_metrics.append({
                        'accuracy': acc, 'balanced_accuracy': bal_acc,
                        'f1': f1, 'auc': auc
                    })
                
                if fold_metrics:
                    # Aggregate results
                    summary = {}
                    for metric in ['accuracy', 'balanced_accuracy', 'f1', 'auc']:
                        values = [m[metric] for m in fold_metrics]
                        summary[metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'values': [float(v) for v in values]
                        }
                    
                    results[model_name] = {'summary': summary}
                    print(f"  ✅ {model_name}: Acc={summary['accuracy']['mean']:.3f} (±{summary['accuracy']['std']:.3f})")
                else:
                    print(f"  ⚠️  {model_name}: No valid folds completed")
            
            except MemoryError:
                print(f"  ❌ {model_name} ran out of memory - model is too large for this dataset")
                continue
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
        
        print_time_estimate(model, args.epochs)
        
        output_file = Path(args.output_dir) / f"results_{model}.json"
        
        cmd = [
            sys.executable, "train_cnn.py",
            "--data_root", args.data_root,
            "--records", *args.records,
            "--cross_validate",
            "--model", model,
            "--epochs", str(args.epochs if not args.quick_test else 25),
            "--batch_size", "64" if model == 'cnn' else "32",  # Larger batch for CNN
            "--output", str(output_file)
        ]
        
        if args.use_smote:
            cmd.append("--use_smote")
        
        print(f"\n  Running: {' '.join(cmd)}\n")
        
        model_start = time.time()
        
        try:
            # Run the training with timeout and output capture
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
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
                print(f"     Accuracy: {acc:.3f}, F1-Score: {f1:.3f}")
        
        except subprocess.TimeoutExpired as e:
            print(f"  ❌ {model.upper()} training timed out after 2 hours")
            print(f"     Consider using --quick_test or reducing --epochs")
            continue
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {model.upper()} failed with error:")
            if e.stderr:
                print(f"     {e.stderr[:500]}")  # First 500 chars of error
            else:
                print(f"     Return code: {e.returncode}")
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
    table.append("="*80)
    table.append("MODEL COMPARISON SUMMARY")
    table.append("="*80)
    table.append(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'AUC':<10} {'Bal. Acc':<10}")
    table.append("-"*80)
    
    # Classical models
    if classical_results:
        table.append("\nCLASSICAL MACHINE LEARNING:")
        table.append("-"*80)
        
        for model_name, results in classical_results.items():
            if 'summary' in results and 'accuracy' in results['summary']:
                acc = results['summary']['accuracy']['mean']
                acc_std = results['summary']['accuracy']['std']
                f1 = results['summary']['f1']['mean']
                f1_std = results['summary']['f1']['std']
                auc = results['summary']['auc']['mean']
                bal_acc = results['summary']['balanced_accuracy']['mean']
                
                table.append(f"{model_name.upper():<25} "
                           f"{acc:.3f}±{acc_std:.3f} {f1:.3f}±{f1_std:.3f} "
                           f"{auc:.3f}      {bal_acc:.3f}")
    
    # Deep learning models
    if dl_results:
        table.append("\nDEEP LEARNING:")
        table.append("-"*80)
        
        for model_name, results in dl_results.items():
            if 'summary' in results and 'accuracy' in results['summary']:
                acc = results['summary']['accuracy']['mean']
                acc_std = results['summary']['accuracy']['std']
                f1 = results['summary']['f1']['mean']
                f1_std = results['summary']['f1']['std']
                auc = results['summary']['auc']['mean']
                bal_acc = results['summary']['balanced_accuracy']['mean']
                
                display_name = model_name.upper()
                if 'training_time' in results:
                    time_min = results['total_training_time_minutes']
                    display_name += f" ({time_min:.0f}m)"
                
                table.append(f"{display_name:<25} "
                           f"{acc:.3f}±{acc_std:.3f} {f1:.3f}±{f1_std:.3f} "
                           f"{auc:.3f}      {bal_acc:.3f}")
    
    table.append("="*80)
    
    return "\n".join(table)


def plot_comparison(classical_results: Dict, dl_results: Dict, output_dir: Path):
    """Generate comparison plots."""
    
    models = []
    accuracies = []
    f1_scores = []
    aucs = []
    model_types = []
    
    # Classical models
    for model_name, results in classical_results.items():
        if 'summary' in results and 'accuracy' in results['summary']:
            models.append(model_name.upper())
            accuracies.append(results['summary']['accuracy']['mean'])
            f1_scores.append(results['summary']['f1']['mean'])
            aucs.append(results['summary']['auc']['mean'])
            model_types.append('Classical ML')
    
    # Deep learning models
    for model_name, results in dl_results.items():
        if 'summary' in results and 'accuracy' in results['summary']:
            display_name = model_name.upper().replace('_', '-')
            models.append(display_name)
            accuracies.append(results['summary']['accuracy']['mean'])
            f1_scores.append(results['summary']['f1']['mean'])
            aucs.append(results['summary']['auc']['mean'])
            model_types.append('Deep Learning')
    
    if not models:
        print("  ⚠️  No data available for plotting")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ECG Arrhythmia Classification: Model Comparison', 
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
    
    # Plot 2: F1-Score
    ax2 = axes[0, 1]
    bars2 = ax2.barh(models, f1_scores, color=colors, alpha=0.8)
    ax2.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model F1-Score', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1])
    for i, v in enumerate(f1_scores):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 3: AUC
    ax3 = axes[1, 0]
    bars3 = ax3.barh(models, aucs, color=colors, alpha=0.8)
    ax3.set_xlabel('ROC-AUC', fontsize=12, fontweight='bold')
    ax3.set_title('Model ROC-AUC', fontsize=13, fontweight='bold')
    ax3.set_xlim([0, 1])
    for i, v in enumerate(aucs):
        ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 4: Combined
    ax4 = axes[1, 1]
    x = np.arange(len(models))
    width = 0.25
    
    bars_acc = ax4.bar(x - width, accuracies, width, label='Accuracy', color='#3498db', alpha=0.8)
    bars_f1 = ax4.bar(x, f1_scores, width, label='F1-Score', color='#2ecc71', alpha=0.8)
    bars_auc = ax4.bar(x + width, aucs, width, label='AUC', color='#f39c12', alpha=0.8)
    
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
    
    report.append("\n## Experimental Setup")
    report.append(f"\n- **Dataset:** MIT-BIH Arrhythmia Database")
    report.append(f"- **Records:** {', '.join(args.records)}")
    report.append(f"- **SMOTE:** {'Yes' if args.use_smote else 'No'}")
    report.append(f"- **Epochs:** {args.epochs}")
    report.append(f"- **LSTM Type:** {'Standard (Slow)' if args.use_slow_lstm else 'Fast (Optimized)'}")
    
    report.append("\n## Results Summary")
    
    # Classical ML
    if classical_results:
        report.append("\n### Classical Machine Learning")
        report.append("\n| Model | Accuracy | F1-Score | AUC | Bal. Accuracy |")
        report.append("|-------|----------|----------|-----|---------------|")
        
        for model_name, results in classical_results.items():
            if 'summary' in results:
                s = results['summary']
                report.append(
                    f"| {model_name.upper()} | "
                    f"{s.get('accuracy', {}).get('mean', 0):.3f} ± {s.get('accuracy', {}).get('std', 0):.3f} | "
                    f"{s.get('f1', {}).get('mean', 0):.3f} ± {s.get('f1', {}).get('std', 0):.3f} | "
                    f"{s.get('auc', {}).get('mean', 0):.3f} ± {s.get('auc', {}).get('std', 0):.3f} | "
                    f"{s.get('balanced_accuracy', {}).get('mean', 0):.3f} ± {s.get('balanced_accuracy', {}).get('std', 0):.3f} |"
                )
    
    # Deep Learning
    if dl_results:
        report.append("\n### Deep Learning")
        report.append("\n| Model | Accuracy | F1-Score | AUC | Training Time |")
        report.append("|-------|----------|----------|-----|---------------|")
        
        for model_name, results in dl_results.items():
            if 'summary' in results:
                s = results['summary']
                time_str = f"{results.get('total_training_time_minutes', 0):.0f}m"
                report.append(
                    f"| {model_name.upper()} | "
                    f"{s.get('accuracy', {}).get('mean', 0):.3f} ± {s.get('accuracy', {}).get('std', 0):.3f} | "
                    f"{s.get('f1', {}).get('mean', 0):.3f} ± {s.get('f1', {}).get('std', 0):.3f} | "
                    f"{s.get('auc', {}).get('mean', 0):.3f} ± {s.get('auc', {}).get('std', 0):.3f} | "
                    f"{time_str} |"
                )
    
    report.append("\n## Visualizations")
    report.append("\n![Model Comparison](model_comparison.png)")
    
    report.append("\n## Notes")
    if not args.use_slow_lstm:
        report.append("\n- Used optimized `fast_lstm` instead of standard LSTM for faster training")
        report.append("- Fast LSTM downsamples input by 4x, reducing training time by 4-8x")
    
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
        print("  - Skipping hybrid CNN-LSTM")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Records: {', '.join(args.records)}")
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
        
        # Generate report
        try:
            generate_detailed_report(classical_results, dl_results, output_dir, args)
        except Exception as e:
            print(f"  ⚠️  Could not generate report: {e}")
        
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