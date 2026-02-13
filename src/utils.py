import re
import numpy as np

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def normalize_percentile(data, p_low=0.5, p_high=99.5):
    if data is None or data.size == 0: return data
    data_f = data.astype(np.float32)
    if data.ndim == 3:
        sample = data_f[::max(1, data.shape[0]//10), ::10, ::10]
    else:
        sample = data_f[::10, ::10]
        
    low, high = np.percentile(sample, [p_low, p_high])
    if high <= low: return np.zeros_like(data, dtype=np.uint8)
    
    norm = (data_f - low) / (high - low)
    return (np.clip(norm, 0, 1) * 255).astype(np.uint8)