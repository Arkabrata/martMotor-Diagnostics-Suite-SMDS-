def get_window_indices(n_samples: int, fs: int, win_sec: float = 1.0):
    win_size = int(fs * win_sec)
    indices = []
    total_windows = n_samples // win_size
    
    for i in range(total_windows):
        start = i * win_size
        end = start + win_size
        indices.append((start, end))
    
    return indices, win_size
