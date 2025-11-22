import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import pandas as pd

CHANNELS = ["acc_x", "acc_y", "acc_z", "mic"]

def extract_features_from_window(window: pd.DataFrame, fs: int):
    feats = {}
    
    for ch in CHANNELS:
        sig = window[ch].values
        prefix = f"{ch}_"

        # Time domain
        feats[prefix+"mean"] = sig.mean()
        feats[prefix+"std"] = sig.std()
        feats[prefix+"rms"] = np.sqrt(np.mean(sig**2))
        feats[prefix+"min"] = sig.min()
        feats[prefix+"max"] = sig.max()
        feats[prefix+"skew"] = skew(sig)
        feats[prefix+"kurt"] = kurtosis(sig)
        feats[prefix+"ptp"] = np.ptp(sig)   # FIX for NumPy 2.0
        feats[prefix+"crest"] = feats[prefix+"max"] / (feats[prefix+"rms"] + 1e-8)

        # Frequency domain
        fft_vals = np.abs(rfft(sig))
        freqs = rfftfreq(len(sig), 1/fs)

        feats[prefix+"dom_freq"] = freqs[np.argmax(fft_vals)]
        feats[prefix+"spec_centroid"] = float(
            np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)
        )
    
    return feats
