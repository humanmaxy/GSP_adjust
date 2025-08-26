from typing import Tuple, Optional
import numpy as np


def create_windows_1d(
    x: np.ndarray,
    window_size: int,
    hop_size: int,
    apply_hann: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice a 1D or 2D array into overlapping windows.

    x: shape (T,) or (C, T)
    returns (windows, window_fn) where windows has shape (N, C, W) and window_fn shape (W,)
    """
    if x.ndim == 1:
        x = x[None, :]
    assert x.ndim == 2, "x must be (T,) or (C, T)"
    C, T = x.shape
    if T < window_size:
        raise ValueError("Signal shorter than window_size")
    if hop_size <= 0:
        raise ValueError("hop_size must be > 0")

    num = 1 + (T - window_size) // hop_size
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(window_size,), axis=1)
    windows = windows[:, ::hop_size, 0, :]
    # windows shape: (C, N, W) -> (N, C, W)
    windows = np.transpose(windows, (1, 0, 2)).copy()

    if apply_hann:
        window_fn = np.hanning(window_size).astype(np.float32)
    else:
        window_fn = np.ones((window_size,), dtype=np.float32)

    return windows.astype(np.float32), window_fn


def overlap_add_reconstruct(
    windows: np.ndarray,
    hop_size: int,
    length: Optional[int] = None,
    window_fn: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct a signal from windows using overlap-add with optional window function compensation.

    windows: shape (N, C, W)
    returns: shape (C, T)
    """
    assert windows.ndim == 3, "windows must be (N, C, W)"
    N, C, W = windows.shape
    T = hop_size * (N - 1) + W
    if length is not None:
        T = length

    out = np.zeros((C, T), dtype=np.float32)
    weight = np.zeros((T,), dtype=np.float32)
    if window_fn is None:
        window_fn = np.ones((W,), dtype=np.float32)

    for i in range(N):
        start = i * hop_size
        end = start + W
        w = window_fn
        out[:, start:end] += windows[i] * w[None, :]
        weight[start:end] += w

    weight = np.maximum(weight, 1e-6)
    out = out / weight[None, :]
    return out


def normalize_minmax(x: np.ndarray, min_val: float, max_val: float, eps: float = 1e-6) -> np.ndarray:
    return 2.0 * (x - min_val) / (max_val - min_val + eps) - 1.0


def denormalize_minmax(y: np.ndarray, min_val: float, max_val: float, eps: float = 1e-6) -> np.ndarray:
    return (y + 1.0) * 0.5 * (max_val - min_val + eps) + min_val