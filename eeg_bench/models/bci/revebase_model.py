from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, cast

import mne
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

class ReveBaseDataset(Dataset):
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

class ReveBaseModel(AbstractModel):
    def __init__(
        self,
        model_name: str = "reve-base-local",
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 2e-4,
        weight_decay: float = 2e-4,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
    ):
        super().__init__("ReveBaseModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = AutoModel.from_pretrained(
            "brain-bzh/reve-base",
            trust_remote_code=True,
            token=True
        ).to(self.device)
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            token=True
        )

        # final_layer in reve-base arch
        dim = embedding_dim if embedding_dim is not None else 45056
        self.model.final_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.RMSNorm(dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(dim, num_classes),
        ).to(self.device)
        
        self.task_name: Optional[str] = None
        self.num_classes: Optional[int] = None

    def _make_positions(self, ch_names: List[str]) -> torch.Tensor:
        pos = self.pos_bank(ch_names)
        pos_bank_chans = self.pos_bank.config.position_names

        missing_chans = [ch for ch in ch_names if ch not in pos_bank_chans]

        if isinstance(pos, (tuple, list)):
            pos = pos[0]
        return pos, missing_chans

    def _select_channels_and_pos(self, data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, torch.Tensor]:
        selected_data = data.astype(np.float32)
        pos, missing_chans = self._make_positions(ch_names)
        if missing_chans:
            print(f"Warning: The following channels are missing from the position bank and will be ignored: {missing_chans}")
            keep_idxs = [i for i, ch in enumerate(ch_names) if ch not in missing_chans]
            selected_data = selected_data[:, keep_idxs, :]
        return selected_data, pos

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

        print(f"Resampling from {orig_freq} Hz to {target_freq} Hz")

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
            data = self.normalize(data)
            labels = np.array([map_label(label, self.task_name) for label in y_], dtype=np.int64)
            dataset = ReveBaseDataset(data, labels, pos)
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
            data = self.normalize(data)
            dataset = ReveBaseDataset(data, None, pos)
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
