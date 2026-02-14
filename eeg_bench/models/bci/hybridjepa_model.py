from __future__ import annotations

import mne

from eeg_bench.models.bci.hybridjepa.hybridjepa import HybridJEPAWithClassifier

from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
import os
from dotenv import load_dotenv
from huggingface_hub import login

from ..abstract_model import AbstractModel
from .LaBraM.utils_2 import calc_class_weights, map_label, n_unique_labels, reverse_map_label

from tqdm import tqdm

# # STANDARD_JEPA_CHANNELS = [
#     "Fp1",
#     "Fp2",
#     "F7",
#     "F3",
#     "Fz",
#     "F4",
#     "F8",
#     "T3",
#     "C3",
#     "Cz",
#     "C4",
#     "T4",
#     "T5",
#     "P3",
#     "Pz",
#     "P4",
#     "T6",
#     "O1",
#     "O2",
# ]

# JEPA_PATCH_SIZE = 200

HYBRID_JEPA_POS =  {
    "FP1": (-3.09, 11.46, 2.79), "FP2": (2.84, 11.53, 2.77), "F7": (-7.19, 7.31, 2.58), "F3": (-5.18, 8.67, 7.87), "FZ": (-0.12, 9.33, 10.26), "F4": (5.03, 8.74, 7.73), "F8": (7.14, 7.45, 2.51), "T3": (-8.60, 1.49, 3.12), "C3": (-6.71, 2.34, 10.45), "CZ": (-0.14, 2.76, 14.02), "C4": (6.53, 2.36, 10.37), "T4": (8.33, 1.53, 3.10), "T5": (-8.77, 1.29, -0.77), "P3": (-5.50, -4.42, 9.99), "PZ": (-0.17, -4.52, 12.67), "P4": (5.36, -4.43, 10.05), "T6": (8.37, 1.17, -0.77), "O1": (-3.16, -8.06, 5.48), "O2": (2.77, -8.05, 5.47),
    # chat gpt
    "Fz":  (-0.12,  9.33, 10.26),

    "FC3": (-5.95,  5.50,  9.16),
    "FC1": (-3.04,  5.78, 10.65),
    "FCz": (-0.13,  6.04, 12.14),
    "FC2": ( 2.83,  5.80, 10.60),
    "FC4": ( 5.78,  5.55,  9.05),

    "C5":  (-7.65,  1.92,  6.79),
    "C3":  (-6.71,  2.34, 10.45),
    "C1":  (-3.42,  2.55, 12.23),
    "Cz":  (-0.14,  2.76, 14.02),
    "C2":  ( 3.20,  2.56, 12.20),
    "C4":  ( 6.53,  2.36, 10.37),
    "C6":  ( 7.43,  1.94,  6.73),

    "CP3": (-6.11, -1.04, 10.22),
    "CP1": (-1.79,  0.83, 12.79),
    "CPz": (-0.16, -0.88, 13.34),
    "CP2": ( 1.52,  0.84, 12.77),
    "CP4": ( 5.95, -1.03, 10.21),

    "P1":  (-2.83, -4.47, 11.33),
    "Pz":  (-0.17, -4.52, 12.67),
    "P2":  ( 2.60, -4.47, 11.36),

    "POz": (-0.18, -6.29,  9.07),

    "Fpz": (-0.12, 11.50, 2.78),

    "AF3": (-4.14, 10.07, 5.33),
    "AF4": ( 3.94, 10.14, 5.25),

    "F5":  (-6.18,  7.99, 5.22),
    "F1":  (-2.65,  9.00, 9.06),
    "F2":  ( 2.46,  9.04, 9.00),
    "F6":  ( 6.08,  8.10, 5.12),

    "FT7": (-7.90,  4.40, 2.85),
    "FC5": (-6.95,  4.82, 6.52),
    "FC6": ( 6.84,  4.90, 6.44),
    "FT8": ( 7.74,  4.49, 2.80),

    # 10-20 aliases (your anchors use T3/T4/T5/T6)
    "T7":  (-8.60,  1.49, 3.12),   # = T3
    "T8":  ( 8.33,  1.53, 3.10),   # = T4

    "TP7": (-7.14, -1.56, 4.61),
    "CP5": (-6.58, -1.25, 8.39),
    "CP6": ( 6.40, -1.24, 8.39),
    "TP8": ( 6.86, -1.63, 4.64),

    "P7":  (-6.48, -2.71, 6.76),
    "P5":  (-5.99, -3.56, 8.38),
    "P6":  ( 5.81, -3.59, 8.43),
    "P8":  ( 6.26, -2.75, 6.80),

    "PO7": (-4.82, -5.38, 6.12),
    "PO5": (-4.58, -5.81, 6.93),
    "PO3": (-4.33, -6.24, 7.74),
    "PO4": ( 4.07, -6.24, 7.76),
    "PO6": ( 4.29, -5.82, 6.95),
    "PO8": ( 4.51, -5.40, 6.14),

    "Oz":  (-0.20, -8.06, 5.48),
}

HYBRID_JEPA_POS = {ch.upper(): pos for ch, pos in HYBRID_JEPA_POS.items()}


class EEGEmbedBaseDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None, pos: Optional[torch.Tensor] = None):
        self.data = data
        self.labels = labels
        self.pos = pos

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.labels is None:
            if self.pos is None:
                return x
            return x, self.pos
        y = int(self.labels[idx])
        if self.pos is None:
            return x, y
        return x, y, self.pos

class HybridJEPAModel(AbstractModel):
    def __init__(
        self,
        checkpoint_path: str = "/share/sv7577-h200-41/checkpoints/HybridJEPA/epoch_10.pth",
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 2e-5,
        weight_decay: float = 2e-4,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
    ):
        super().__init__("HybridJEPAModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = HybridJEPAWithClassifier(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            freeze=True
        ).to(self.device)

        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            token=True
        )
        
        self.task_name: Optional[str] = None
        self.num_classes: Optional[int] = None

    def _make_positions(self, meta) -> torch.Tensor:
        # get position from montage assuming 1020
        ch_names = meta["channel_names"]
        print(ch_names)
        missing = [ch for ch in ch_names if ch.upper() not in HYBRID_JEPA_POS]
        if missing:
            print(f"Warning: Missing positions for channels {missing}. These channels will be ignored.")
        return torch.tensor([HYBRID_JEPA_POS[ch.upper()] for ch in ch_names if ch.upper() in HYBRID_JEPA_POS], dtype=torch.float32)

    # def _adjust_channels(
    #     self,
    #     data: np.ndarray,
    #     ch_names: List[str],
    #     target_channels: int = 19,
    # ) -> Tuple[np.ndarray, List[str]]:
    #     if len(ch_names) >= target_channels:
    #         return data[:, :target_channels, :], ch_names[:target_channels]

    #     ch_upper = [ch.upper() for ch in ch_names]
    #     missing = [ch for ch in STANDARD_JEPA_CHANNELS if ch.upper() not in ch_upper]
    #     needed = target_channels - len(ch_names)
    #     pad_names = missing[:needed]
    #     pad_data = np.zeros((data.shape[0], needed, data.shape[2]), dtype=data.dtype)
    #     padded_data = np.concatenate([data, pad_data], axis=1)
    #     padded_names = ch_names + pad_names
    #     return padded_data, padded_names

    def _select_channels_and_pos(self, data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, torch.Tensor]:
        # shave off the last timepoint for each channel
        # FIXME - why do we have to do this?
        # data = data[:, :, :-1]

        # data, ch_names = self._adjust_channels(data, ch_names)
        selected_data = data.astype(np.float32)
        pos = self._make_positions({"channel_names": ch_names})
        return selected_data, pos

    # def _adjust_time_length(self, data: np.ndarray, patch_size: int = JEPA_PATCH_SIZE) -> np.ndarray:
    #     time_len = data.shape[2]
    #     remainder = time_len % patch_size
    #     if remainder == 0:
    #         return data
    #     target_len = time_len - remainder
    #     if target_len == 0:
    #         raise ValueError(
    #             f"Time length {time_len} is shorter than patch size {patch_size} after trimming."
    #         )
    #     return data[:, :, :target_len]

    def _forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos.expand(x.shape[0], -1, -1)
        out = self.model(x, pos.to(self.device))

        return out

    def normalize(self, data: np.ndarray) -> np.ndarray:
        # assuming b x c x t
        data_mean = data.mean(axis=(0, 2), keepdims=True)
        data_std = data.std(axis=(0, 2), keepdims=True) + 1e-6
        normalized = (data - data_mean) / data_std
        return normalized
    
    def _resample(self, data, meta, target_freq=200):
        orig_freq = meta["sampling_frequency"]
        if orig_freq == target_freq:
            return data

        # print(f"Resampling from {orig_freq} Hz to {target_freq} Hz")

        data_float = data.astype(np.float64, copy=False)

        if orig_freq < target_freq:
            # upsample
            resampled = mne.filter.resample(data_float, up=target_freq / orig_freq, verbose=False)
        else:
            # downsample
            resampled = mne.filter.resample(data_float, down=orig_freq / target_freq, verbose=False)

        return resampled.astype(np.float32, copy=False)


    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        self.task_name = meta[0]["task_name"]
        self.num_classes = n_unique_labels(self.task_name)

        class_weights = torch.tensor(calc_class_weights(y, self.task_name), dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        datasets = []
        loaders = []
        for X_, y_, meta_ in zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta):
            if X_.ndim != 3 or X_.size == 0:
                continue
            data, pos = self._select_channels_and_pos(X_, meta_["channel_names"])
            data = self._resample(data, meta_)
            # data = self._adjust_time_length(data)
            data = self.normalize(data)
            labels = np.array([map_label(label, self.task_name) for label in y_], dtype=np.int64)
            dataset = EEGEmbedBaseDataset(data, labels, pos)
            datasets.append(dataset)
            loaders.append(DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0))

        if not loaders:
            return

        with torch.no_grad():
            for loader in loaders:
                for batch in loader:
                    xb, _, posb = batch
                    xb = xb.to(self.device)
                    _ = self._forward(xb, posb)
                    break
                break

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        for _ in range(self.epochs):
            for loader in loaders:
                for xb, yb, posb in tqdm(loader):
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self._forward(xb, posb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        if self.task_name is None or self.num_classes is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        self.model.eval()

        predictions = []
        for X_, meta_ in zip(cast(List[np.ndarray], X), meta):
            if X_.ndim != 3 or X_.size == 0:
                continue
            data, pos = self._select_channels_and_pos(X_, meta_["channel_names"])
            data = self._resample(data, meta_)
            # data = self._adjust_time_length(data)
            data = self.normalize(data)
            dataset = EEGEmbedBaseDataset(data, None, pos)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            batch_preds = []
            for xb, posb in tqdm(loader):
                xb = xb.to(self.device)
                logits = self._forward(xb, posb)
                preds = torch.argmax(logits, dim=1)
                batch_preds.append(preds.cpu().numpy())

            if not batch_preds:
                continue

            flat_preds = np.concatenate(batch_preds, axis=0)
            mapped = np.array([reverse_map_label(int(p), self.task_name) for p in flat_preds])
            predictions.append(mapped)

        if not predictions:
            return np.array([])
        return np.concatenate(predictions, axis=0)
