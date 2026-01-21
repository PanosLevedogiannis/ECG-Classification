# ECG Arrhythmia Classification System

A complete machine learning pipeline for automated ECG arrhythmia detection using classical ML and deep learning.

## 📋 What This Does

Classifies ECG signals as **Normal** or **Arrhythmia** using two approaches:
- **Classical ML**: SVM, Random Forest, Gradient Boosting
- **Deep Learning**: CNN, LSTM, ResNet with skip connections

Uses real data from the MIT-BIH Arrhythmia Database and prevents overfitting through patient-wise cross-validation.

## 🎯 Quick Start (5 minutes)

### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data (Required)

```bash
# Create data directory and download MIT-BIH database
mkdir -p data/mitdb
cd data/mitdb

# Download 20 patient records (you need the .dat, .hea, and .atr files)
# Option A: Using wget (Linux/Mac)
for i in 100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121; do
    wget https://physionet.org/files/mitdb/1.0.0/$i.dat
    wget https://physionet.org/files/mitdb/1.0.0/$i.hea
    wget https://physionet.org/files/mitdb/1.0.0/$i.atr
done

# Option B: Manual download
# Visit: https://physionet.org/content/mitdb/1.0.0/
# Download the 3 files (.dat, .hea, .atr) for each patient above
# Place them in data/mitdb/

cd ../..
```

### 3. Run Your First Comparison

```bash
# Test with classical ML models (fastest, ~5 minutes)
python compare_models.py --skip_deep --quick_test

# Expected output:
# ✅ Classical ML Summary:
#    Random Forest: Balanced Accuracy ~0.80
#    SVM: Balanced Accuracy ~0.75
#    Gradient Boosting: Balanced Accuracy ~0.78
```

Done! You'll find results in `results/` directory.

## 📖 Common Usage Scenarios

### Scenario 1: Train Only Classical ML Models (Fastest)

```bash
python compare_models.py --skip_deep --use_smote
```

**Pros:**
- Trains in 5-10 minutes
- No GPU needed
- Easy to interpret results

**Output:** `results/all_results.json`, comparison plots

### Scenario 2: Train Only Deep Learning (ResNet)

```bash
python train_cnn.py --cross_validate --model cnn --use_smote --epochs 50
```

**Pros:**
- State-of-the-art ResNet with skip connections
- Better for complex patterns
- Uses Focal Loss for hard example learning

**Output:** `results_deep_learning.json`

### Scenario 3: Full Comparison (All Models)

```bash
python compare_models.py --use_smote
```

**Takes:** ~30-60 minutes depending on CPU/GPU
**Output:** Complete comparison table and plots

### Scenario 4: Quick Test (5 minutes)

```bash
python compare_models.py --quick_test
```

**Pros:**
- Tests CNN and Fast LSTM only
- 25 epochs instead of 50
- Good for debugging

## 🛠️ Installation Details

### Prerequisites
- Python 3.8+
- pip or conda
- ~2 GB disk space for data
- Optional: NVIDIA GPU for faster deep learning training

### Step-by-Step Installation

**1. Clone and setup:**
```bash
cd /your/project/directory
python -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Verify installation:**
```bash
python -c "import tensorflow; import wfdb; import sklearn; print('✅ All dependencies installed')"
```

**4. Download data (see Section 2 above)**

## 📊 Understanding the Results

### What You'll See

After running `python compare_models.py --skip_deep --quick_test`, you'll get:

```
CLASSICAL MACHINE LEARNING (with fixes)
─────────────────────────────────────────────
Model              Train Loss    Test Loss    Bal.Acc    F1      AUC
SVM                0.3721        0.4285       0.753±0.100  0.71   0.821±0.065
Random Forest      0.0015        0.2134       0.801±0.085  0.76   0.851±0.052
Gradient Boosting  0.0821        0.1923       0.782±0.092  0.74   0.834±0.058
```

**Key Metrics Explained:**
- **Balanced Accuracy**: Average of sensitivity and specificity (best for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve (1.0 = perfect, 0.5 = random)
- **Train Loss vs Test Loss Gap**: Indicates overfitting (should be < 0.3)

### Output Files

After running, you'll find in `results/`:
- `all_results.json` - Complete numerical results
- `model_comparison.png` - Comparison plots
- `report.md` - Detailed markdown report
- `results_*.json` - Individual model results

## ⚠️ Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'wfdb'"

**Solution:**
```bash
pip install wfdb
```

### Problem: "Data not found" or "No valid windows extracted"

**Solution:** Verify data is in correct location:
```bash
ls -la data/mitdb/100.*  # Should show 100.dat, 100.hea, 100.atr
```

### Problem: Out of Memory (OOM) Error

**Solution:** Reduce batch size or records:
```bash
python train_cnn.py --batch_size 16 --records 100 101 102
```

### Problem: Training is very slow

**Solution:** Make sure you're using `fast_lstm`:
```bash
python compare_models.py  # Uses fast_lstm by default ✅

# DON'T use this (4-8x slower):
# python compare_models.py --use_slow_lstm  # ❌ Very slow!
```

### Problem: SMOTE not available

**Solution:**
```bash
pip install imbalanced-learn
```

### Problem: Wavelet features disabled

**Solution:**
```bash
pip install PyWavelets
```

## 📁 Project Structure

```
ecg-arrhythmia-classification/
├── compare_models.py          ← Run this first! (all models)
├── train_cnn.py               ← Deep learning training
├── ecg_mitbih.py              ← Signal processing (don't modify)
├── requirements.txt           ← Dependencies
├── data/
│   └── mitdb/                 ← Put data files here
│       ├── 100.dat
│       ├── 100.hea
│       ├── 100.atr
│       └── ... (more patient files)
├── results/                   ← Output directory (auto-created)
│   ├── all_results.json
│   ├── model_comparison.png
│   └── report.md
└── README.md
```

## 🚀 Advanced Usage

### Use Different ECG Records

```bash
# Use only records 100-105
python compare_models.py --skip_deep --records 100 101 102 103 104 105

# Use all available records
python compare_models.py --skip_deep --records 100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121
```

### Enable Data Augmentation

```bash
# Deep learning with augmentation
python train_cnn.py --cross_validate --use_augmentation --epochs 50
```

### Add Wavelet Features (Classical ML)

```bash
# Classical ML with wavelet features (more features but slower)
python compare_models.py --skip_deep --include_wavelet
```

### Custom Output Directory

```bash
python compare_models.py --output_dir my_results
```

### Use Gradient Boosting + SMOTE Only

```bash
# XGBoost with multiprocessing (n_jobs=4 for speed)
python compare_models.py --skip_deep --skip_classical --records 100
```

## 📈 Expected Performance

### With Default Settings (20 patients, SMOTE)

| Model | Balanced Accuracy | F1-Score | AUC |
|-------|-------------------|----------|-----|
| SVM | 75.3% ± 10.0% | 71% | 0.821 |
| Random Forest | 80.1% ± 8.5% | 76% | 0.851 |
| CNN (ResNet) | 72.0% ± 8.0% | 68% | 0.790 |
| Fast LSTM | 70.5% ± 9.2% | 66% | 0.775 |

*Note: Results vary based on data and random seed*

## 🎓 What You'll Learn

This project demonstrates:

1. **Signal Processing**: Filtering, feature extraction, frequency analysis
2. **Machine Learning**: Classification, cross-validation, hyperparameter tuning
3. **Deep Learning**: CNN, LSTM, ResNet architectures
4. **Best Practices**: Patient-wise CV, class balancing with SMOTE, focal loss
5. **Reproducibility**: Fixed random seeds, detailed logging

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `compare_models.py` | **Main entry point** - trains all classical ML and deep learning models |
| `train_cnn.py` | Deep learning training script (CNN, LSTM, ResNet) |
| `ecg_mitbih.py` | Signal processing & feature extraction (don't edit) |
| `requirements.txt` | Python dependencies |

## 🤔 FAQ

**Q: Which model should I use?**
A: Start with SVM or Random Forest (fast, interpretable). Use deep learning if you need maximum accuracy.

**Q: How long does training take?**
A: Classical ML: ~5-10 min | Deep Learning: ~30-60 min | Full comparison: ~60-120 min

**Q: Do I need a GPU?**
A: No, CPU is fine. GPU (NVIDIA) makes deep learning ~5-10x faster.

**Q: Can I use my own ECG data?**
A: Yes, but you'll need to format it as WFDB records (.dat, .hea, .atr files).

**Q: What if results are poor?**
A: Check data quality, try SMOTE, increase epochs, or use deep learning.

## 📞 Support

If you encounter issues:
1. Check the **Troubleshooting** section above
2. Verify data is in `data/mitdb/`
3. Ensure all dependencies installed: `pip install -r requirements.txt`
4. Try the quick test: `python compare_models.py --quick_test`

## 📄 License & References

**MIT-BIH Database:**
- Moody, G. B., & Mark, R. G. (2001). "The impact of the MIT-BIH Arrhythmia Database." IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
- https://physionet.org/content/mitdb/

**Techniques Used:**
- Focal Loss (Lin et al., 2017)
- ResNet-1D (Deep skip connections)
- SMOTE (Chawla et al., 2002)

---

**Ready to start?** Run:
```bash
python compare_models.py --quick_test
```

## 📖 Usage

### 1. Train Classical ML Models

Classical ML training is integrated into `compare_models.py`:

```bash
# Train classical models only
python compare_models.py --skip_deep --use_smote

# Train all classical models with wavelet features
python compare_models.py --skip_deep --use_smote --include_wavelet

# Quick test with fewer records
python compare_models.py --skip_deep --records 100 101 102 103 104
```

**Options:**
- `--skip_deep`: Skip deep learning models (classical only)
- `--use_smote`: Apply SMOTE oversampling
- `--include_wavelet`: Add wavelet features (requires pywt)
- `--quick_test`: Reduce epochs and data for testing
- `--records`: Specify which records to use

### 2. Train Deep Learning Models


### 2. Train Deep Learning Models

```bash
# Train 1D CNN
python train_cnn.py --cross_validate --model cnn --use_smote

# Train Fast LSTM (recommended - 4-8x faster)
python train_cnn.py --cross_validate --model fast_lstm --epochs 50

# Train CNN-LSTM hybrid
python train_cnn.py --cross_validate --model cnn_lstm --use_focal_loss

# Full configuration
python train_cnn.py \
    --cross_validate \
    --model cnn \
    --use_smote \
    --use_focal_loss \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0003
```

**Options:**
- `--model [cnn|lstm|fast_lstm|cnn_lstm]`: Choose architecture
- `--use_smote`: Apply SMOTE oversampling
- `--use_focal_loss`: Use focal loss instead of cross-entropy
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for Adam optimizer

### 3. Compare All Models

```bash
# Run complete comparison (recommended)
python compare_models.py --use_smote --include_wavelet

# Quick test mode (faster, fewer epochs)
python compare_models.py --quick_test

# Skip classical or deep learning
python compare_models.py --skip_classical  # Only deep learning
python compare_models.py --skip_deep       # Only classical ML

# Use optimized LSTM (recommended)
python compare_models.py  # Uses fast_lstm by default

# Use slow standard LSTM (NOT recommended)
python compare_models.py --use_slow_lstm  # Takes 4-8x longer!

# Custom configuration
python compare_models.py \
    --use_smote \
    --include_wavelet \
    --epochs 50 \
    --output_dir results_final \
    --records 100 101 102 103 104 105 106 107 108 109
```

**Options:**
- `--use_smote`: Apply SMOTE oversampling for all models
- `--include_wavelet`: Add wavelet features for classical models
- `--epochs`: Number of training epochs for deep learning
- `--quick_test`: Fast mode (25 epochs, CNN + Fast LSTM only)
- `--skip_classical`: Skip classical ML models
- `--skip_deep`: Skip deep learning models
- `--use_slow_lstm`: Use standard LSTM instead of fast_lstm (NOT RECOMMENDED)
- `--output_dir`: Directory to save results

**Output:**
- `results/results_rf.json`: Random Forest results
- `results/results_svm.json`: SVM results
- `results/results_cnn.json`: CNN results
- `results/results_fast_lstm.json`: Fast LSTM results
- `results/results_cnn_lstm.json`: CNN-LSTM results
- `results/model_comparison.png`: Comparison plots
- `results/report.md`: Detailed markdown report

### Performance Tips

- **Use GPU for deep learning**: Install `tensorflow-gpu`
- **⚠️ IMPORTANT**: Use `--model fast_lstm` instead of `lstm`
  - Standard LSTM is 4-8x slower on long sequences (1800 timesteps)
  - Fast LSTM downsamples input by 4x with minimal accuracy loss
- **Reduce records for quick testing**: `--records 100 101 102 103 104`
- **Quick test mode**: `python compare_models.py --quick_test`
- **Skip slow models**: `python compare_models.py --skip_deep` or `--skip_classical`
- **Use parallel processing**: scikit-learn uses all cores by default
- **Reduce batch size if OOM**: `--batch_size 16`

## 📁 Project Structure

```
ecg-arrhythmia-classification/
├── ecg_mitbih.py              # Core signal processing & feature extraction
├── train_cnn.py               # Deep learning training pipeline (CNN/LSTM/CNN-LSTM)
├── compare_models.py          # Comprehensive model comparison (Classical + Deep)
├── requirements.txt           # Python dependencies
├── data/
│   └── mitdb/                 # MIT-BIH database files
│       ├── 100.dat
│       ├── 100.hea
│       ├── 100.atr
│       └── ...
├── results/                   # Output directory
│   ├── results_rf.json       # Random Forest results
│   ├── results_svm.json      # SVM results
│   ├── results_cnn.json      # CNN results
│   ├── results_fast_lstm.json # Fast LSTM results
│   ├── results_cnn_lstm.json # CNN-LSTM results
│   ├── model_comparison.png  # Comparison plots
│   └── report.md             # Detailed markdown report
└── README.md
```

## 🔬 Methodology

### Signal Processing Pipeline

1. **Preprocessing**
   - Load ECG records from MIT-BIH database
   - Apply Butterworth bandpass filter (0.5-40 Hz)
   - Remove baseline wander and high-frequency noise
   - **⚠️ Important**: Data is standardized using StandardScaler (zero mean, unit variance)
     - Bandpass filtering removes frequency components but does NOT standardize amplitude
     - Neural networks require standardized inputs for stable training
     - Classical models benefit from feature normalization

2. **Segmentation**
   - Create 5-second windows with 2.5-second overlap
   - Label based on beat annotations (Normal vs Arrhythmia)

3. **Feature Extraction**
   - **Time Domain**: Statistical features (mean, std, skewness, kurtosis, RMS, etc.)
   - **Frequency Domain**: FFT-based power spectral density, band powers, spectral entropy
   - **HRV Features**: R-R intervals, RMSSD, pNN50, heart rate statistics
   - **Wavelet Domain** (optional): Multi-level DWT coefficients and energy distribution

### Machine Learning Pipeline

1. **Classical ML**
   - Extract engineered features (40-70 dimensions)
   - Normalize using StandardScaler
   - Apply SMOTE if class imbalance exists
   - Train SVM, Random Forest, Gradient Boosting, Logistic Regression
   - Patient-wise 5-fold cross-validation

2. **Deep Learning**
   - Use raw signal (1800 timesteps × 2 channels)
   - Normalize using StandardScaler
   - Apply SMOTE if requested
   - Train CNN/LSTM/CNN-LSTM architectures
   - Stratified validation split (80/20)
   - Patient-wise 5-fold cross-validation

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Average of per-class accuracies
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **ROC-AUC**: Area under ROC curve

## 📈 Expected Results

Based on literature and preliminary experiments:

### Classical ML
- **SVM**: Accuracy ~75-85%, F1 ~70-80%
- **Random Forest**: Accuracy ~80-88%, F1 ~75-83%
- **Gradient Boosting**: Accuracy ~78-86%, F1 ~73-81%

### Deep Learning
- **1D CNN**: Accuracy ~85-92%, F1 ~82-89% (⚡ Fast: ~10-15 min)
- **Fast BiLSTM**: Accuracy ~83-90%, F1 ~80-87% (⚡ Fast: ~15-20 min)
- **BiLSTM**: Accuracy ~83-90%, F1 ~80-87% (🐌 SLOW: ~60-120 min)
- **CNN-LSTM**: Accuracy ~87-93%, F1 ~84-90% (⚡ Fast: ~20-30 min)

*Note: Training times are for 5-fold CV with 50 epochs on 48 patient records. Results vary based on patient distribution and class imbalance*

## ⚠️ Common Pitfalls & Solutions

### 1. **Using Standard LSTM Instead of Fast LSTM**
**Problem**: Training takes hours instead of minutes
**Solution**: Always use `--model fast_lstm` unless you specifically need standard LSTM

### 2. **Skipping StandardScaler**
**Problem**: Poor model performance, unstable training
**Solution**: Data is automatically standardized (don't skip this step)

### 3. **Not Using Enough Patient Records**
**Problem**: Extreme class imbalance, unreliable cross-validation
**Solution**: Use at least 20+ patient records (default: 48 records)

### 4. **Forgetting Cross-Subject Validation**
**Problem**: Overoptimistic results due to data leakage
**Solution**: Patient-wise CV is automatic (patients in test ≠ patients in train)

### 5. **Class Imbalance Issues**
**Problem**: Model only predicts majority class
**Solutions**:
- Use `--use_smote` for oversampling
- Use `--use_focal_loss` for deep learning
- Check balanced_accuracy metric (more reliable than accuracy)

### 6. **Memory Errors**
**Problem**: OOM errors during training
**Solutions**:
- Reduce `--batch_size` (try 16 or 8)
- Use fewer `--records`
- Use `fast_lstm` instead of `lstm`
- Close other applications

## 🎓 Theoretical Background

### ΣΕΣ (Signal Processing)

The ECG signal processing pipeline implements fundamental concepts:

1. **Digital Filtering**
   - Butterworth filter: Maximally flat passband
   - Removes power line interference (50/60 Hz)
   - Eliminates muscle noise and baseline wander

2. **Frequency Analysis**
   - Fourier Transform: Time → Frequency domain
   - Power Spectral Density: Energy distribution across frequencies
   - Important for detecting abnormal frequency components

3. **Wavelet Transform**
   - Time-frequency representation
   - Captures transient events (QRS complexes, arrhythmias)
   - Multi-resolution analysis

### ΜΜ (Machine Learning)

The classification pipeline demonstrates key ML concepts:

1. **Feature Engineering**
   - Transforms raw signals into meaningful features
   - Domain knowledge guides feature selection
   - Critical for classical ML performance

2. **Classification**
   - **SVM**: Maximum margin separation in high-dimensional space
   - **Random Forest**: Ensemble of decision trees
   - **CNN**: Automatic feature learning through convolution
   - **LSTM**: Long-term temporal dependencies

3. **Cross-Validation**
   - Patient-wise splitting: Ensures generalization
   - Prevents data leakage
   - Realistic performance estimation

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: wfdb**
   ```bash
   pip install wfdb
   ```

2. **SMOTE not available**
   ```bash
   pip install imbalanced-learn
   ```

3. **Wavelet features disabled**
   ```bash
   pip install PyWavelets
   ```

4. **TensorFlow errors**
   ```bash
   pip install --upgrade tensorflow
   ```

5. **Memory errors with deep learning**
   - Reduce batch size: `--batch_size 16`
   - Use fewer records
   - Increase system swap space

### Performance Tips

- **Use GPU for deep learning**: Install `tensorflow-gpu`
- **Reduce records for quick testing**: `--records 100 101 102`
- **Skip wavelet features** for faster classical ML
- **Use parallel processing**: scikit-learn uses all cores by default

## 📚 References

### Datasets
- MIT-BIH Arrhythmia Database: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

### Papers
1. Lin et al. "Focal Loss for Dense Object Detection" (2017)
2. Hannun et al. "Cardiologist-level arrhythmia detection" (2019)
3. Rajpurkar et al. "Deep learning for chest radiograph diagnosis" (2017)

### Tools
- WFDB Python Package: [Documentation](https://wfdb.readthedocs.io/)
- scikit-learn: [User Guide](https://scikit-learn.org/stable/)
- TensorFlow/Keras: [Tutorials](https://www.tensorflow.org/tutorials)

## 👥 Contributors

- Panagiotis Leventogiannis - Initial work

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- MIT-BIH Arrhythmia Database creators
- PhysioNet community
- Open-source ML/DL frameworks

---

**Note**: This is an academic project for biomedical signal processing course. Not intended for clinical use.
