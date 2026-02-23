from __future__ import annotations

from typing import Dict, List, Optional, Tuple, cast

import mne
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

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


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5.0))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B) * self.scaling
        return base_out + lora_out


class LoRAQKVLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        if base_layer.out_features % 3 != 0:
            raise ValueError("Expected packed QKV projection with out_features divisible by 3")

        self.base = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        qkv_dim = self.base.out_features // 3
        in_dim = self.base.in_features

        self.lora_A_q = nn.Parameter(torch.empty(self.rank, in_dim))
        self.lora_A_k = nn.Parameter(torch.empty(self.rank, in_dim))
        self.lora_A_v = nn.Parameter(torch.empty(self.rank, in_dim))

        self.lora_B_q = nn.Parameter(torch.zeros(qkv_dim, self.rank))
        self.lora_B_k = nn.Parameter(torch.zeros(qkv_dim, self.rank))
        self.lora_B_v = nn.Parameter(torch.zeros(qkv_dim, self.rank))

        nn.init.kaiming_uniform_(self.lora_A_q, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_A_k, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_A_v, a=np.sqrt(5.0))
        nn.init.zeros_(self.lora_B_q)
        nn.init.zeros_(self.lora_B_k)
        nn.init.zeros_(self.lora_B_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_d = self.dropout(x)

        q = F.linear(F.linear(x_d, self.lora_A_q), self.lora_B_q)
        k = F.linear(F.linear(x_d, self.lora_A_k), self.lora_B_k)
        v = F.linear(F.linear(x_d, self.lora_A_v), self.lora_B_v)

        lora_out = torch.cat([q, k, v], dim=-1) * self.scaling
        return base_out + lora_out


class ReveBaseModel(AbstractModel):
    def __init__(
        self,
        model_name: str = "reve-base-local",
        batch_size: int = 64,
        epochs: int = 30,
        lr: float = 3e-5,
        weight_decay: float = 2e-4,
        embedding_dim: Optional[int] = None,
        num_classes: int = 2,
        linear_probe_epochs: int = 3,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.1,
        mixup_alpha: float = 0.4,
        plateau_factor: float = 0.5,
        plateau_patience: int = 3,
        plateau_min_lr: float = 1e-6,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_only_finetune: bool = True,
    ):
        super().__init__("ReveBaseModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.linear_probe_epochs = max(0, linear_probe_epochs)
        self.warmup_epochs = max(0, warmup_epochs)
        self.warmup_start_factor = float(warmup_start_factor)
        self.mixup_alpha = float(mixup_alpha)
        self.plateau_factor = float(plateau_factor)
        self.plateau_patience = int(plateau_patience)
        self.plateau_min_lr = float(plateau_min_lr)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_only_finetune = bool(lora_only_finetune)
        self.lora_enabled = False

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        def load_repo(repo_id: str):
            local = snapshot_download(repo_id=repo_id, token=hf_token)
            return AutoModel.from_pretrained(local, trust_remote_code=True, local_files_only=True)

        self.model = load_repo("brain-bzh/reve-base")
        self.pos_bank = load_repo("brain-bzh/reve-positions")
        self._inject_lora_into_attention()

        dim = embedding_dim if embedding_dim is not None else 45056
        self.model.final_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.RMSNorm(dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(dim, num_classes),
        )
        self.model = self.model.to(self.device)

        self.task_name: Optional[str] = None
        self.num_classes: Optional[int] = None

    def _set_two_step_trainable(self, train_backbone: bool) -> None:
        for name, param in self.model.named_parameters():
            if name.startswith("final_layer"):
                param.requires_grad = True
                continue

            is_lora_param = "lora_" in name
            if not train_backbone:
                param.requires_grad = False
            elif self.lora_enabled and self.lora_only_finetune:
                param.requires_grad = is_lora_param
            else:
                param.requires_grad = train_backbone

    def _inject_lora_into_attention(self) -> None:
        if self.lora_rank <= 0:
            self.lora_enabled = False
            return

        attention_layers = 0
        for module in self.model.modules():
            if hasattr(module, "to_qkv") and isinstance(module.to_qkv, nn.Linear):
                if not isinstance(module.to_qkv, LoRAQKVLinear):
                    module.to_qkv = LoRAQKVLinear(
                        module.to_qkv,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )
                attention_layers += 1

            if hasattr(module, "to_out") and isinstance(module.to_out, nn.Linear):
                if not isinstance(module.to_out, LoRALinear):
                    module.to_out = LoRALinear(
                        module.to_out,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )

        self.lora_enabled = attention_layers > 0
        if self.lora_rank > 0 and not self.lora_enabled:
            print("Warning: LoRA requested but no attention projections were found.")

    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.mixup_alpha <= 0.0 or x.shape[0] < 2:
            return x, y, y, 1.0
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        index = torch.randperm(x.shape[0], device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _set_warmup_lr(self, optimizer: torch.optim.Optimizer, epoch_idx: int) -> None:
        if self.warmup_epochs <= 0 or epoch_idx >= self.warmup_epochs:
            return
        warmup_progress = float(epoch_idx + 1) / float(self.warmup_epochs)
        lr_scale = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * warmup_progress
        lr_scale = min(1.0, max(0.0, lr_scale))
        for group in optimizer.param_groups:
            group["lr"] = self.lr * lr_scale

    def _make_positions(self, ch_names: List[str]) -> Tuple[torch.Tensor, List[str]]:
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
            selected_data = selected_data[:, keep_ids, :]
        return selected_data, pos

    def _forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos.expand(x.shape[0], -1, -1)
        out = self.model(x, pos.to(self.device))

        return out

    def normalize(self, data: np.ndarray) -> np.ndarray:
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
            resampled = mne.filter.resample(data_float, up=target_freq / orig_freq, verbose=False)
        else:
            resampled = mne.filter.resample(data_float, down=orig_freq / target_freq, verbose=False)

        return resampled.astype(np.float32, copy=False)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        self.task_name = meta[0]["task_name"]
        self.num_classes = n_unique_labels(self.task_name)

        class_weights = torch.tensor(calc_class_weights(y, self.task_name), dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        loaders = []
        for X_, y_, meta_ in zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta):
            if X_.ndim != 3 or X_.size == 0:
                continue
            data, pos = self._select_channels_and_pos(X_, meta_["channel_names"])
            data = self._resample(data, meta_)
            data = self.normalize(data)
            labels = np.array([map_label(label, self.task_name) for label in y_], dtype=np.int64)
            dataset = ReveBaseDataset(data, labels, pos)
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
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.plateau_factor,
            patience=self.plateau_patience,
            min_lr=self.plateau_min_lr,
        )

        self._set_two_step_trainable(train_backbone=False)
        stage2_start = min(max(0, self.linear_probe_epochs), self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            if epoch == stage2_start:
                self._set_two_step_trainable(train_backbone=True)
            self._set_warmup_lr(optimizer, epoch)

            epoch_loss_sum = 0.0
            epoch_batches = 0
            for loader in loaders:
                for xb, yb, posb in tqdm(loader):
                    xb = xb.to(self.device)
                    yb = yb.to(self.device).long()
                    xb, y_a, y_b, lam = self._apply_mixup(xb, yb)
                    optimizer.zero_grad()
                    logits = self._forward(xb, posb)
                    if lam < 1.0:
                        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                    else:
                        loss = criterion(logits, y_a)
                    loss.backward()
                    optimizer.step()
                    epoch_loss_sum += float(loss.detach().item())
                    epoch_batches += 1

            if epoch >= self.warmup_epochs and epoch_batches > 0:
                scheduler.step(epoch_loss_sum / float(epoch_batches))

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
