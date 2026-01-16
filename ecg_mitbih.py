"""
ECG Signal Processing and Feature Extraction Module
Author: Advanced Biomedical Signal Processing System
Version: 2.0

This module provides comprehensive feature extraction for ECG signals including:
- Time domain features
- Frequency domain features (FFT-based)
- Heart Rate Variability (HRV) features
- Wavelet-based features
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

# Optional: Wavelet transform (install with: pip install PyWavelets)
try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    warnings.warn("PyWavelets not installed. Wavelet features will be disabled.")

# MIT-BIH: Normal beat annotations
NORMAL_BEATS = {"N", "L", "R", "e", "j"}


def bandpass_filter(
    sig: np.ndarray, 
    fs: float, 
    low: float = 0.5, 
    high: float = 40.0
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to ECG signal.
    
    Args:
        sig: Input signal (n_samples,) or (n_samples, n_channels)
        fs: Sampling frequency in Hz
        low: Low cutoff frequency in Hz
        high: High cutoff frequency in Hz
    
    Returns:
        Filtered signal with same shape as input
    """
    nyquist = fs / 2
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    b, a = butter(4, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, sig, axis=0)


def load_record(data_root: str | Path, rec_name: str):
    """Load a single ECG record with annotations."""
    base = Path(data_root) / rec_name
    record = wfdb.rdrecord(str(base))
    ann = wfdb.rdann(str(base), "atr")
    return record, ann


def extract_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical time-domain features from signal.
    
    Args:
        signal: 1D array of signal values
    
    Returns:
        Dictionary of time-domain features
    """
    features = {
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'var': float(np.var(signal)),
        'min': float(np.min(signal)),
        'max': float(np.max(signal)),
        'range': float(np.ptp(signal)),
        'median': float(np.median(signal)),
        'mad': float(np.median(np.abs(signal - np.median(signal)))),  # Median Absolute Deviation
        'rms': float(np.sqrt(np.mean(signal**2))),  # Root Mean Square
        'skewness': float(skew(signal)),
        'kurtosis': float(kurtosis(signal)),
        'peak_to_peak': float(np.ptp(signal)),
    }
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(signal)))
    features['zero_crossing_rate'] = float(zero_crossings / len(signal))
    
    return features


def extract_frequency_domain_features(
    signal: np.ndarray, 
    fs: float
) -> Dict[str, float]:
    """
    Extract frequency-domain features using FFT and Welch's method.
    
    Args:
        signal: 1D array of signal values
        fs: Sampling frequency in Hz
    
    Returns:
        Dictionary of frequency-domain features
    """
    # Welch's method for PSD estimation (more robust than raw FFT)
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    
    # Define frequency bands (ECG-specific)
    bands = {
        'vlf': (0.0, 0.04),   # Very Low Frequency
        'lf': (0.04, 0.15),    # Low Frequency
        'hf': (0.15, 0.4),     # High Frequency
        'vhf': (0.4, 40.0),    # Very High Frequency
    }
    
    features = {}
    total_power = np.trapz(psd, freqs)
    
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        
        features[f'power_{band_name}'] = float(band_power)
        features[f'power_{band_name}_ratio'] = float(band_power / (total_power + 1e-10))
    
    # LF/HF ratio (important for HRV analysis)
    lf_power = features['power_lf']
    hf_power = features['power_hf']
    features['lf_hf_ratio'] = float(lf_power / (hf_power + 1e-10))
    
    # Spectral features
    features['total_power'] = float(total_power)
    features['dominant_freq'] = float(freqs[np.argmax(psd)])
    features['mean_freq'] = float(np.sum(freqs * psd) / np.sum(psd))
    features['median_freq'] = float(np.median(freqs[psd > np.median(psd)]))
    
    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-10)
    features['spectral_entropy'] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
    
    return features


def extract_hrv_features(signal: np.ndarray, fs: float, verbose: bool = False) -> Optional[Dict[str, float]]:
    """
    Extract Heart Rate Variability (HRV) features.
    
    Args:
        signal: 1D array of ECG signal
        fs: Sampling frequency in Hz
        verbose: Print warnings on extraction failures
    
    Returns:
        Dictionary of HRV features or None if insufficient peaks found
    """
    # R-peak detection
    # Use adaptive threshold based on signal statistics
    threshold = 0.5 * np.std(signal)
    distance = int(fs * 0.6)  # Minimum 600ms between beats (100 bpm max)
    
    peaks, properties = find_peaks(
        signal,
        distance=distance,
        prominence=threshold,
        height=np.mean(signal) + 0.3 * np.std(signal)
    )
    
    if len(peaks) < 3:
        if verbose:
            print(f"    ⚠️  HRV extraction failed: only {len(peaks)} R-peaks detected (need ≥3)")
        return None
    
    # RR intervals in milliseconds
    rr_intervals = np.diff(peaks) / fs * 1000
    
    if len(rr_intervals) == 0:
        if verbose:
            print(f"    ⚠️  HRV extraction failed: no RR intervals computed")
        return None
    
    features = {}
    
    # Time-domain HRV features
    features['hrv_mean_rr'] = float(np.mean(rr_intervals))
    features['hrv_std_rr'] = float(np.std(rr_intervals))
    features['hrv_rmssd'] = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))
    features['hrv_sdsd'] = float(np.std(np.diff(rr_intervals)))
    
    # Heart rate statistics
    hr = 60000 / rr_intervals
    features['hrv_mean_hr'] = float(np.mean(hr))
    features['hrv_std_hr'] = float(np.std(hr))
    features['hrv_min_hr'] = float(np.min(hr))
    features['hrv_max_hr'] = float(np.max(hr))
    
    # RR interval range features
    features['hrv_min_rr'] = float(np.min(rr_intervals))
    features['hrv_max_rr'] = float(np.max(rr_intervals))
    features['hrv_range_rr'] = float(np.ptp(rr_intervals))
    
    # NN50 and pNN50
    if len(rr_intervals) > 1:
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        features['hrv_nn50'] = float(nn50)
        features['hrv_pnn50'] = float((nn50 / len(rr_intervals)) * 100)
    
    # Geometric features
    features['hrv_num_peaks'] = len(peaks)
    features['hrv_peak_density'] = float(len(peaks) / (len(signal) / fs))  # peaks per second
    
    # Coefficient of variation
    features['hrv_cv'] = float(features['hrv_std_rr'] / (features['hrv_mean_rr'] + 1e-10))
    
    return features


def extract_wavelet_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract wavelet-based features using Discrete Wavelet Transform.
    
    Args:
        signal: 1D array of signal values
        fs: Sampling frequency in Hz
    
    Returns:
        Dictionary of wavelet features
    """
    if not WAVELET_AVAILABLE:
        return {}
    
    features = {}
    
    # Multi-level DWT decomposition
    wavelet = 'db4'  # Daubechies 4
    level = 5
    
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Extract features from each level
        for i, coeff in enumerate(coeffs):
            prefix = f'wavelet_d{i}' if i > 0 else 'wavelet_a'
            
            features[f'{prefix}_energy'] = float(np.sum(coeff ** 2))
            features[f'{prefix}_mean'] = float(np.mean(np.abs(coeff)))
            features[f'{prefix}_std'] = float(np.std(coeff))
            features[f'{prefix}_max'] = float(np.max(np.abs(coeff)))
        
        # Energy distribution
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = sum(energies)
        
        for i, energy in enumerate(energies):
            prefix = f'wavelet_d{i}' if i > 0 else 'wavelet_a'
            features[f'{prefix}_energy_ratio'] = float(energy / (total_energy + 1e-10))
    
    except (ValueError, RuntimeError, MemoryError) as e:
        warnings.warn(f"Wavelet feature extraction failed: {e}")
    
    return features


def extract_all_features(
    signal: np.ndarray, 
    fs: float,
    include_wavelet: bool = True
) -> Dict[str, float]:
    """
    Extract all available features from a signal window.
    
    Args:
        signal: 1D or 2D array (if 2D, uses first channel)
        fs: Sampling frequency in Hz
        include_wavelet: Whether to include wavelet features
    
    Returns:
        Dictionary containing all extracted features
    """
    # Handle multi-channel signals
    if signal.ndim > 1:
        signal = signal[:, 0]
    
    features = {}
    
    # Time domain features
    time_features = extract_time_domain_features(signal)
    features.update(time_features)
    
    # Frequency domain features
    freq_features = extract_frequency_domain_features(signal, fs)
    features.update(freq_features)
    
    # HRV features
    hrv_features = extract_hrv_features(signal, fs, verbose=False)
    if hrv_features is not None:
        features.update(hrv_features)
    else:
        # Fill with zeros if HRV extraction failed
        dummy_hrv = {
            'hrv_mean_rr': 0.0, 'hrv_std_rr': 0.0, 'hrv_rmssd': 0.0,
            'hrv_mean_hr': 0.0, 'hrv_std_hr': 0.0, 'hrv_min_hr': 0.0,
            'hrv_max_hr': 0.0, 'hrv_min_rr': 0.0, 'hrv_max_rr': 0.0,
            'hrv_range_rr': 0.0, 'hrv_nn50': 0.0, 'hrv_pnn50': 0.0,
            'hrv_num_peaks': 0.0, 'hrv_peak_density': 0.0, 'hrv_cv': 0.0,
            'hrv_sdsd': 0.0
        }
        features.update(dummy_hrv)
    
    # Wavelet features (optional)
    if include_wavelet and WAVELET_AVAILABLE:
        wavelet_features = extract_wavelet_features(signal, fs)
        features.update(wavelet_features)
    
    return features


def make_windows_from_record(
    record,
    ann,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    bandpass: Optional[Tuple[float, float]] = (0.5, 40.0),
    extract_features: bool = False,
    include_wavelet: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed-length windows from an ECG record.
    
    Args:
        record: WFDB record object
        ann: WFDB annotation object
        window_sec: Window length in seconds
        step_sec: Step size in seconds (overlap control)
        bandpass: Tuple of (low, high) cutoff frequencies or None
        extract_features: If True, extract features instead of raw signal
        include_wavelet: Whether to include wavelet features
    
    Returns:
        X: (n_windows, features) array
        y: (n_windows,) label array (0=Normal, 1=Arrhythmia)
    """
    fs = record.fs
    sig = record.p_signal.astype(float)
    n_samples, n_ch = sig.shape

    if bandpass is not None:
        sig = bandpass_filter(sig, fs, low=bandpass[0], high=bandpass[1])

    win = int(window_sec * fs)
    step = int(step_sec * fs)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    beat_samples = np.asarray(ann.sample)
    beat_symbols = np.asarray(ann.symbol)

    for start in range(0, n_samples - win + 1, step):
        end = start + win
        
        # Check for beats in window
        mask = (beat_samples >= start) & (beat_samples < end)
        if not mask.any():
            continue

        window = sig[start:end, :]
        
        # Determine label
        symbols_in_win = set(beat_symbols[mask])
        label = 0 if all(s in NORMAL_BEATS for s in symbols_in_win) else 1

        if extract_features:
            # Extract feature vector
            feature_dict = extract_all_features(window, fs, include_wavelet)
            feature_vec = np.array(list(feature_dict.values()), dtype=np.float32)
            X_list.append(feature_vec)
        else:
            # Ensure consistent shape handling for raw signal
            if window.ndim == 1:
                # If window is 1D, use as-is
                X_list.append(window.astype(np.float32))
            else:
                # If window is 2D, use first channel (Lead II) only
                X_list.append(window[:, 0].astype(np.float32))
        
        y_list.append(label)

    if not X_list:
        feature_dim = len(extract_all_features(sig[:win], fs, include_wavelet)) if extract_features else (n_ch * win)
        return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=int)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y


def load_dataset(
    data_root: str | Path,
    records: List[str],
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    bp: Optional[Tuple[float, float]] = (0.5, 40.0),
    use_features: bool = False,
    include_wavelet: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load multiple ECG records and combine into a single dataset.
    
    Args:
        data_root: Path to data directory
        records: List of record names to load
        window_sec: Window length in seconds
        step_sec: Step size in seconds
        bp: Bandpass filter cutoffs (low, high) or None
        use_features: If True, extract features; if False, use raw signal
        include_wavelet: Whether to include wavelet features
        verbose: Print progress information
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        groups: Record identifier for each sample (n_samples,)
    """
    X_all, y_all, groups_all = [], [], []
    
    for rec in records:
        if verbose:
            print(f"Processing record {rec} ...")
        
        try:
            record, ann = load_record(data_root, rec)
            X, y = make_windows_from_record(
                record, ann, 
                window_sec, 
                step_sec, 
                bandpass=bp,
                extract_features=use_features,
                include_wavelet=include_wavelet
            )
            
            if X.shape[0] == 0:
                if verbose:
                    print(f"  ⚠️  No valid windows extracted from {rec}")
                continue
            
            X_all.append(X)
            y_all.append(y)
            groups_all.extend([rec] * X.shape[0])
            
            if verbose:
                mode = "features" if use_features else "raw signal"
                print(f"  ✓ {rec}: {X.shape[0]} windows, "
                      f"{X.shape[1]} {mode} dimensions, "
                      f"arrhythmia ratio={y.mean():.3f}")
        
        except Exception as e:
            print(f"  ✗ Error processing {rec}: {e}")
            continue
    
    if not X_all:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=int), np.empty((0,), dtype=object)
    
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups = np.array(groups_all, dtype=object)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset loaded successfully!")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(f"  Class distribution: Normal={np.sum(y==0)}, Arrhythmia={np.sum(y==1)}")
        print(f"  Unique patients: {len(np.unique(groups))}")
        print(f"{'='*60}\n")
    
    return X, y, groups


def get_feature_names(include_wavelet: bool = True) -> List[str]:
    """
    Get the names of all features in order.
    
    Args:
        include_wavelet: Whether wavelet features are included
    
    Returns:
        List of feature names
    """
    # Create dummy signal to extract features
    dummy_signal = np.random.randn(1800)
    feature_dict = extract_all_features(dummy_signal, fs=360, include_wavelet=include_wavelet)
    return list(feature_dict.keys())