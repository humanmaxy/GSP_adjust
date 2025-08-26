import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class GyroTempWindowDataset(Dataset):
    def __init__(
        self,
        files_glob: str,
        window_size: int,
        hop_size: int,
        require_multiple_of: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.files: List[str] = sorted(glob.glob(files_glob))
        if len(self.files) == 0:
            raise ValueError(f"No files matched: {files_glob}")
        self.window_size = window_size
        self.hop_size = hop_size
        self.require_multiple_of = require_multiple_of

        # Build index over all files and windows
        self._index: List[Tuple[int, int]] = []  # (file_idx, window_idx)
        self._file_stats: List[Tuple[float, float, float, float]] = []  # (gyro_min, gyro_max, temp_min, temp_max)
        self._file_lengths: List[int] = []
        self._file_data_cache: List[Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]] = []

        for fi, f in enumerate(self.files):
            data = np.load(f)
            gyro = data.get("gyro")
            temp = data.get("temp")
            target = data.get("target")
            if gyro is None or temp is None:
                raise ValueError(f"File {f} missing 'gyro' or 'temp'")
            if target is None:
                target = gyro.copy()

            gyro = np.asarray(gyro).astype(np.float32).reshape(-1)
            temp = np.asarray(temp).astype(np.float32).reshape(-1)
            target = np.asarray(target).astype(np.float32).reshape(-1)

            T = min(len(gyro), len(temp), len(target))
            gyro = gyro[:T]
            temp = temp[:T]
            target = target[:T]

            if require_multiple_of is not None and T % require_multiple_of != 0:
                # Trim tail to nearest multiple
                T_trim = T - (T % require_multiple_of)
                if T_trim < window_size:
                    # Skip this file if too short after trimming
                    continue
                gyro = gyro[:T_trim]
                temp = temp[:T_trim]
                target = target[:T_trim]
                T = T_trim

            gyro_min, gyro_max = float(np.min(gyro)), float(np.max(gyro))
            temp_min, temp_max = float(np.min(temp)), float(np.max(temp))
            self._file_stats.append((gyro_min, gyro_max, temp_min, temp_max))
            self._file_lengths.append(T)
            self._file_data_cache.append((gyro, temp, target))

            if T < window_size:
                continue
            num_windows = 1 + (T - window_size) // hop_size
            for wi in range(num_windows):
                self._index.append((fi, wi))

        if len(self._index) == 0:
            raise ValueError("No windows constructed; check data lengths and window/hop sizes")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fi, wi = self._index[idx]
        gyro, temp, target = self._file_data_cache[fi]  # type: ignore[index]
        assert gyro is not None and temp is not None and target is not None

        W = self.window_size
        start = wi * self.hop_size
        end = start + W

        g = gyro[start:end]
        t = temp[start:end]
        y = target[start:end]

        gmin, gmax, tmin, tmax = self._file_stats[fi]
        # Normalize to [-1, 1]
        eps = 1e-6
        g_norm = 2.0 * (g - gmin) / (gmax - gmin + eps) - 1.0 if gmax > gmin else g - gmin
        t_norm = 2.0 * (t - tmin) / (tmax - tmin + eps) - 1.0 if tmax > tmin else t - tmin
        y_norm = 2.0 * (y - gmin) / (gmax - gmin + eps) - 1.0 if gmax > gmin else y - gmin

        x = np.stack([g_norm, t_norm], axis=0)  # (2, W)
        yx = y_norm[None, :]  # (1, W)
        return torch.from_numpy(x).float(), torch.from_numpy(yx).float()