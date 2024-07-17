import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample, butter, filtfilt


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if self.transform:
            self.X[i] = self.transform(self.X[i])
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]


class PreprocessedMEGDataset(ThingsMEGDataset):
    def __init__(self, split, data_dir="data", resampling_rate=False, baseline_correction=True, scaling=True, low_pass_filter=False, noise_augmentation=False):
        super().__init__(split, data_dir)
        self.resampling_rate = resampling_rate
        self.baseline_correction = baseline_correction
        self.scaling = scaling
        self.low_pass_filter = low_pass_filter
        self.noise_augmentation = noise_augmentation
        self.original_rate = 1000  # 仮定する元のサンプリングレート

    def __getitem__(self, i):
        X = self.X[i]
        subject_idx = self.subject_idxs[i]

        # リサンプリング
        if self.resampling_rate:
            X = self.resample_data(X, self.resampling_rate)

        # ローパスフィルタリング
        if self.low_pass_filter:
            X = self.apply_low_pass_filter(X)

        # ベースライン補正
        if self.baseline_correction:
            X = self.correct_baseline(X)

        # スケーリング
        if self.scaling:
            X = self.scale_data(X)

        # ガウシアンノイズの追加
        if self.noise_augmentation:
            X = self.add_gaussian_noise(X)

        if hasattr(self, "y"):
            y = self.y[i]
            return X, y, subject_idx
        else:
            return X, subject_idx

    def resample_data(self, data, new_rate):
        original_len = data.shape[-1]
        num_samples = int(original_len * new_rate / self.original_rate)
        resampled_data = torch.tensor(resample(data.numpy(), num_samples), dtype=torch.float32)
        return resampled_data

    def apply_low_pass_filter(self, data, cutoff=30, fs=100, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data.numpy())
        return torch.tensor(filtered_data.copy(), dtype=torch.float32)  # 配列をコピーしてからTensorに変換


    def correct_baseline(self, data):
        baseline = data.mean(dim=-1, keepdim=True)
        corrected_data = data - baseline
        return corrected_data

    def scale_data(self, data):
        min_val = data.min()
        max_val = data.max()
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data

    def add_gaussian_noise(self, data, mean=0.0, std=0.01):
        noise = torch.randn(data.size()) * std + mean
        noisy_data = data + noise
        return noisy_data

