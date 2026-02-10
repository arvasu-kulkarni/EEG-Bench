from __future__ import annotations

import mne

from eeg_bench.models.bci.jepa.jepa import EEG_JEPA_MAE_Classifier

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

STANDARD_JEPA_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]

JEPA_PATCH_SIZE = 200

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

class JEPAModel(AbstractModel):
    def __init__(
        self,
        checkpoint_path: str = "/share/sv7577-h200-41/checkpoints/MAE-JEPA/epoch_12.ckpt",
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 2e-5,
        weight_decay: float = 2e-4,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
    ):
        super().__init__("JEPAModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = EEG_JEPA_MAE_Classifier(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
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
        mne_raw_info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        raw_obj = mne.io.RawArray(np.zeros((len(ch_names), 100)), mne_raw_info)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_obj.set_montage(montage, match_case=False, on_missing="raise")
        pos = torch.tensor(list(raw_obj.get_montage().get_positions()["ch_pos"].values()), dtype=torch.float32)
        scaled_pos = 100 * pos
        # print(scaled_pos)
        return scaled_pos

    def _adjust_channels(
        self,
        data: np.ndarray,
        ch_names: List[str],
        target_channels: int = 19,
    ) -> Tuple[np.ndarray, List[str]]:
        if len(ch_names) >= target_channels:
            return data[:, :target_channels, :], ch_names[:target_channels]

        ch_upper = [ch.upper() for ch in ch_names]
        missing = [ch for ch in STANDARD_JEPA_CHANNELS if ch.upper() not in ch_upper]
        needed = target_channels - len(ch_names)
        pad_names = missing[:needed]
        pad_data = np.zeros((data.shape[0], needed, data.shape[2]), dtype=data.dtype)
        padded_data = np.concatenate([data, pad_data], axis=1)
        padded_names = ch_names + pad_names
        return padded_data, padded_names

    def _select_channels_and_pos(self, data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, torch.Tensor]:
        # shave off the last timepoint for each channel
        # FIXME - why do we have to do this?
        data = data[:, :, :-1]

        data, ch_names = self._adjust_channels(data, ch_names)
        selected_data = data.astype(np.float32)
        pos = self._make_positions({"channel_names": ch_names})
        return selected_data, pos

    def _adjust_time_length(self, data: np.ndarray, patch_size: int = JEPA_PATCH_SIZE) -> np.ndarray:
        time_len = data.shape[2]
        remainder = time_len % patch_size
        if remainder == 0:
            return data
        target_len = time_len - remainder
        if target_len == 0:
            raise ValueError(
                f"Time length {time_len} is shorter than patch size {patch_size} after trimming."
            )
        return data[:, :, :target_len]

    def _forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos.expand(x.shape[0], -1, -1)
        out = self.model(x, pos.to(self.device))

        return out
    
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
            data = self._adjust_time_length(data)
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
            data = self._adjust_time_length(data)
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
