from __future__ import annotations

from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, get_channels, map_label_reverse


class ReveBaseModel(AbstractModel):
    def __init__(
        self,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 2e-4,
        weight_decay: float = 2e-4,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
    ):
        super().__init__("ReveBaseModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.use_cache = True

        self.model = AutoModel.from_pretrained(
            "brain-bzh/reve-base",
            trust_remote_code=True,
            token=True,
        ).to(self.device)
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            token=True,
        )

        dim = embedding_dim if embedding_dim is not None else 45056
        self.model.final_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.RMSNorm(dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(dim, num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)),
        ).to(self.device)

        self.task_name: Optional[str] = None
        self.default_channels: Optional[List[str]] = None

    def _make_positions(self, ch_names: List[str]) -> Tuple[torch.Tensor, List[str]]:
        pos = self.pos_bank(ch_names)
        pos_bank_chans = self.pos_bank.config.position_names

        missing_chans = [ch for ch in ch_names if ch not in pos_bank_chans]

        if isinstance(pos, (tuple, list)):
            pos = pos[0]
        if isinstance(pos, torch.Tensor) and pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos[0]
        return pos, missing_chans

    def _select_channels_and_pos(self, data: torch.Tensor, ch_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        pos, missing_chans = self._make_positions(ch_names)
        if missing_chans:
            keep_idxs = [i for i, ch in enumerate(ch_names) if ch not in missing_chans]
            if keep_idxs:
                data = data[:, keep_idxs, :]
        return data, pos

    def _align_time_and_pos(
        self,
        data: torch.Tensor,
        pos: torch.Tensor,
        ch_names: List[str],
        patch_size: int = 200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = data.shape[-1]
        h = t // patch_size
        if h < 1:
            return data, pos
        t_new = h * patch_size
        if t_new != t:
            data = data[..., :t_new]
        expected = data.shape[1] * h

        pos_channel = self.pos_bank(ch_names)
        if isinstance(pos_channel, (tuple, list)):
            pos_channel = pos_channel[0]
        if pos_channel.ndim == 3 and pos_channel.shape[0] == 1:
            pos_channel = pos_channel[0]
        if pos_channel.ndim == 1 and data.shape[1] > 0 and pos_channel.shape[0] % data.shape[1] == 0:
            pos_channel = pos_channel.view(data.shape[1], -1)
        if pos_channel.ndim == 2:
            if pos_channel.shape[0] == 1 and data.shape[1] > 1:
                pos_channel = pos_channel.repeat(data.shape[1], 1)
            elif pos_channel.shape[0] != data.shape[1]:
                embed_dim = pos_channel.shape[1] if pos_channel.shape[0] > 0 else 1
                pos_channel = torch.zeros((data.shape[1], embed_dim), dtype=pos_channel.dtype, device=pos_channel.device)

        pos_full = pos_channel.repeat_interleave(h, dim=0)
        if pos_full.shape[0] != expected:
            raise ValueError(f"Position embedding length {pos_full.shape[0]} does not match expected {expected}")
        return data, pos_full

    def _forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos.expand(x.shape[0], -1, -1)
        return self.model(x, pos.to(self.device))

    def _extract_channels(self, batch_channels) -> List[str]:
        if batch_channels == -1 or batch_channels is None:
            return self.default_channels or []
        if isinstance(batch_channels, (list, tuple)):
            if len(batch_channels) == 0:
                return self.default_channels or []
            first = batch_channels[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return [str(ch).upper() for ch in first]
            if isinstance(first, str):
                return [str(ch).upper() for ch in batch_channels]
        return self.default_channels or []

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        self.task_name = meta[0]["task_name"]
        self.default_channels = get_channels(self.task_name)

        class_weights = torch.tensor(calc_class_weights(y, self.task_name), dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        dataset_train = make_dataset_2(
            X,
            y,
            meta,
            self.task_name,
            self.name,
            self.chunk_len_s,
            is_train=True,
            use_cache=self.use_cache,
        )
        val_split = 0.2
        if val_split is not None:
            dataset_train, dataset_val = dataset_train.split_train_val(val_split)
        else:
            dataset_val = None

        if self.chunk_len_s is None:
            batch_size = 1
        else:
            batch_size = self.batch_size
        train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, shuffle=True)
        valid_loader = None
        if dataset_val is not None:
            valid_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=False)

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb, channels in tqdm(train_loader, desc="Training", leave=False):
                ch_names = self._extract_channels(channels)
                if xb.ndim == 2:
                    xb = xb.unsqueeze(0)
                xb = xb.to(self.device, dtype=torch.float32)
                xb, posb = self._select_channels_and_pos(xb, ch_names)
                xb, posb = self._align_time_and_pos(xb, posb, ch_names)
                yb = torch.as_tensor(yb, device=self.device)
                if not self.is_multilabel_task:
                    yb = yb.long()

                optimizer.zero_grad()
                logits = self._forward(xb, posb)
                if self.is_multilabel_task:
                    logits = logits.reshape((logits.shape[0], self.num_classes, -1))
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if valid_loader is None:
                continue
            self.model.eval()
            with torch.no_grad():
                for xb, yb, channels in tqdm(valid_loader, desc="Validation", leave=False):
                    ch_names = self._extract_channels(channels)
                    if xb.ndim == 2:
                        xb = xb.unsqueeze(0)
                    xb = xb.to(self.device, dtype=torch.float32)
                    xb, posb = self._select_channels_and_pos(xb, ch_names)
                    xb, posb = self._align_time_and_pos(xb, posb, ch_names)
                    yb = torch.as_tensor(yb, device=self.device)
                    if not self.is_multilabel_task:
                        yb = yb.long()
                    logits = self._forward(xb, posb)
                    if self.is_multilabel_task:
                        logits = logits.reshape((logits.shape[0], self.num_classes, -1))
                    _ = criterion(logits, yb)
            self.model.train()

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        if self.task_name is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        dataset = make_dataset_2(
            X,
            None,
            meta,
            self.task_name,
            self.name,
            self.chunk_len_s,
            is_train=False,
            use_cache=self.use_cache,
        )
        if self.chunk_len_s is None:
            batch_size = 1
        else:
            batch_size = self.batch_size
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        self.model.eval()
        predictions = []
        for xb, _, channels in tqdm(loader, desc="Testing", leave=False):
            ch_names = self._extract_channels(channels)
            if xb.ndim == 2:
                xb = xb.unsqueeze(0)
            xb = xb.to(self.device, dtype=torch.float32)
            xb, posb = self._select_channels_and_pos(xb, ch_names)
            xb, posb = self._align_time_and_pos(xb, posb, ch_names)
            logits = self._forward(xb, posb)
            if self.is_multilabel_task:
                logits = logits.reshape((logits.shape[0], self.num_classes, -1))
                preds = torch.argmax(logits, dim=1)
                predictions.append(preds.cpu().numpy())
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                mapped = np.array([map_label_reverse(int(p), self.task_name) for p in preds])
                predictions.append(mapped)

        if not predictions:
            return np.array([])
        return np.concatenate(predictions, axis=0)
