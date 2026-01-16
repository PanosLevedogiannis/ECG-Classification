# ECG Arrhythmia Classification System

Advanced biomedical signal processing and machine learning system for automated ECG arrhythmia detection.

## 🎯 Overview

This project implements a comprehensive pipeline for ECG arrhythmia classification using both **classical machine learning** and **deep learning** approaches. It demonstrates the integration of:

- **Signal Processing (ΣΕΣ)**: Digital filtering, frequency analysis, wavelet transforms
- **Machine Learning (ΜΜ)**: Feature engineering, classification, cross-validation

## 📊 Features

### Signal Processing
- ✅ Butterworth bandpass filtering (0.5-40 Hz)
- ✅ Sliding window segmentation with overlap
- ✅ **Time-domain features** (12 features): mean, std, skewness, kurtosis, RMS, etc.
- ✅ **Frequency-domain features** (11 features): Power spectral density, spectral entropy, band powers
- ✅ **HRV features** (17 features): R-R intervals, RMSSD, pNN50, heart rate statistics
- ✅ **Wavelet features** (optional, 30+ features): Multi-level DWT decomposition

### Machine Learning Models

#### Classical ML
- **SVM**: Support Vector Machine with RBF kernel
- **Random Forest**: 200 trees with balanced weights
- **Gradient Boosting**: 100 estimators
- **Logistic Regression**: L2 regularization

#### Deep Learning
- **1D CNN**: 4 convolutional blocks with batch normalization (recommended)
- **Fast BiLSTM**: Optimized bidirectional LSTM with 4x downsampling
- **BiLSTM**: Standard bidirectional LSTM (SLOW, use fast_lstm instead)
- **CNN-LSTM**: Hybrid architecture combining spatial and temporal learning

### Advanced Training Strategies
- ✅ **Patient-wise cross-validation** (5-fold)
- ✅ **SMOTE** oversampling for class imbalance
- ✅ **Focal Loss** for hard example mining
- ✅ **Class weights** for balanced training
- ✅ **Early stopping** and learning rate scheduling
- ✅ **L2 regularization** to prevent overfitting

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo>
cd ecg-arrhythmia-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
wfdb>=4.0.0
tensorflow>=2.8.0
keras>=2.8.0
imbalanced-learn>=0.9.0
PyWavelets>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
```

**Optional GPU Support:**
```bash
# For NVIDIA GPUs (much faster training)
pip install tensorflow-gpu
```

### Data Preparation

1. Download MIT-BIH Arrhythmia Database:
```bash
mkdir -p data/mitdb
cd data/mitdb

# Download using WFDB tools
for i in {100..109}; do
    wget https://physionet.org/files/mitdb/1.0.0/${i}.dat
    wget https://physionet.org/files/mitdb/1.0.0/${i}.hea
    wget https://physionet.org/files/mitdb/1.0.0/${i}.atr
done
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