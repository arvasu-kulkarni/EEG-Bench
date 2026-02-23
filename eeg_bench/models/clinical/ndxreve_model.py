from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import mne
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, get_channels, map_label_reverse


def _canonical_channel_name(name: str) -> str:
    clean = name.upper().strip()
    clean = clean.replace("EEG ", "")
    clean = clean.replace("-REF", "")
    clean = clean.replace("-LE", "")
    clean = clean.replace(" ", "")
    clean = re.sub(r"[^A-Z0-9]", "", clean)
    return clean


class NDXReveClassifier(nn.Module):
    def __init__(self, mae: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = mae.patch_embed
        self.pos_enc = mae.pos_enc
        self.encoder = mae.encoder
        self.patch_size = int(mae.patch_size)
        self.step = int(mae.step)
        self.dropout = float(dropout)
        self.final_layer: Optional[nn.Module] = None
        self.flat_dim: Optional[int] = None

    def prepare_coords(self, xyz: torch.Tensor, num_patches: int) -> torch.Tensor:
        bsz, channels, _ = xyz.shape
        device = xyz.device
        time_idx = torch.arange(num_patches, device=device, dtype=torch.float32)
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)
        time = time_idx.view(1, 1, num_patches, 1).expand(bsz, channels, -1, -1)
        return torch.cat([spat, time], dim=-1).flatten(1, 2)

    def encode(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]
        tokens = self.patch_embed.linear(patches).flatten(1, 2)
        coords = self.prepare_coords(pos, num_patches)
        pe = self.pos_enc(coords)
        latents, _ = self.encoder(tokens + pe)
        return latents

    def init_head(self, latents: torch.Tensor, out_dim: int) -> None:
        flat_dim = int(latents.shape[1] * latents.shape[2])
        if self.final_layer is not None and self.flat_dim == flat_dim:
            return
        self.flat_dim = flat_dim
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(flat_dim),
            nn.Dropout(self.dropout),
            nn.Linear(flat_dim, out_dim),
        ).to(latents.device)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        latents = self.encode(x, pos)
        if self.final_layer is None:
            raise RuntimeError("Classifier head is not initialized.")
        return self.final_layer(latents)


class NdxReveModel(AbstractModel):
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        ndx_model_py_path: str = "/home/neurodx/mahir/ndx-pipeline/model.py",
        fs: int = 200,
        patch_seconds: float = 1.0,
        overlap_seconds: float = 0.1,
        embed_dim: int = 512,
        encoder_depth: int = 12,
        encoder_heads: int = 8,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        mask_ratio: float = 0.55,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 2e-4,
        weight_decay: float = 2e-4,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        linear_probe_epochs: int = 3,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.1,
        mixup_alpha: float = 0.4,
        plateau_factor: float = 0.5,
        plateau_patience: int = 3,
        plateau_min_lr: float = 1e-6,
        dropout: float = 0.1,
    ):
        super().__init__("NdxReveModel")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.num_classes = int(num_classes)
        self.num_labels_per_chunk = num_labels_per_chunk
        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.use_cache = True

        self.linear_probe_epochs = max(0, int(linear_probe_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(warmup_start_factor)
        self.mixup_alpha = float(mixup_alpha)
        self.plateau_factor = float(plateau_factor)
        self.plateau_patience = int(plateau_patience)
        self.plateau_min_lr = float(plateau_min_lr)
        self.target_fs = int(fs)

        checkpoint_path = checkpoint_path or os.getenv("NDX_REVE_CHECKPOINT")
        if not checkpoint_path:
            raise ValueError("Missing checkpoint path. Set --ndx-checkpoint or NDX_REVE_CHECKPOINT.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(ndx_model_py_path):
            raise FileNotFoundError(f"ndx-pipeline model.py not found: {ndx_model_py_path}")

        mae_cls = self._load_mae_class(ndx_model_py_path)
        mae = mae_cls(
            fs=fs,
            patch_seconds=patch_seconds,
            overlap_seconds=overlap_seconds,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            mask_ratio=mask_ratio,
        )
        state_dict = self._load_state_dict(checkpoint_path)
        mae.load_state_dict(state_dict, strict=True)

        self.model = NDXReveClassifier(mae=mae, dropout=dropout).to(self.device)
        self.task_name: Optional[str] = None
        self.default_channels: Optional[List[str]] = None
        self.fixed_num_samples: Optional[int] = None

        montage = mne.channels.make_standard_montage("standard_1020")
        self._montage_ch_pos = montage.get_positions()["ch_pos"]
        self._montage_lookup = {_canonical_channel_name(ch): ch for ch in self._montage_ch_pos.keys()}
        self._pos_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        self._missing_pos_warned: set[Tuple[str, ...]] = set()

    @staticmethod
    def _load_mae_class(model_py_path: str):
        model_path = Path(model_py_path)
        spec = importlib.util.spec_from_file_location("ndx_pipeline_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {model_py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "MAE"):
            raise AttributeError(f"No MAE class found in {model_py_path}")
        return getattr(module, "MAE")

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned[key[len("module."):]] = value
            else:
                cleaned[key] = value
        return cleaned

    def _load_state_dict(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "model", "state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return self._strip_module_prefix(cast(Dict[str, torch.Tensor], ckpt[key]))
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return self._strip_module_prefix(cast(Dict[str, torch.Tensor], ckpt))
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    def _set_two_step_trainable(self, train_backbone: bool) -> None:
        for name, param in self.model.named_parameters():
            if name.startswith("final_layer"):
                param.requires_grad = True
            else:
                param.requires_grad = train_backbone

    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.mixup_alpha <= 0.0 or x.shape[0] < 2:
            return x, y, y, 1.0
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        index = torch.randperm(x.shape[0], device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[index]
        return mixed_x, y, y[index], lam

    def _set_warmup_lr(self, optimizer: torch.optim.Optimizer, epoch_idx: int) -> None:
        if self.warmup_epochs <= 0 or epoch_idx >= self.warmup_epochs:
            return
        warmup_progress = float(epoch_idx + 1) / float(self.warmup_epochs)
        lr_scale = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * warmup_progress
        lr_scale = min(1.0, max(0.0, lr_scale))
        for group in optimizer.param_groups:
            group["lr"] = self.lr * lr_scale

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

    def _align_channels(self, data: torch.Tensor, src_channels: List[str], target_channels: List[str]) -> torch.Tensor:
        if not target_channels:
            return data
        src_map = {_canonical_channel_name(ch): idx for idx, ch in enumerate(src_channels)}
        aligned = torch.zeros((data.shape[0], len(target_channels), data.shape[-1]), dtype=data.dtype, device=data.device)
        for out_idx, target_ch in enumerate(target_channels):
            src_idx = src_map.get(_canonical_channel_name(target_ch))
            if src_idx is not None and 0 <= src_idx < data.shape[1]:
                aligned[:, out_idx, :] = data[:, src_idx, :]
        return aligned

    def _build_positions(self, ch_names: List[str]) -> torch.Tensor:
        key = tuple(ch_names)
        if key in self._pos_cache:
            return self._pos_cache[key]

        pos = []
        missing = []
        for ch in ch_names:
            canonical = _canonical_channel_name(ch)
            base_name = self._montage_lookup.get(canonical)
            if base_name is None:
                pos.append([0.0, 0.0, 0.0])
                missing.append(ch)
            else:
                coord = self._montage_ch_pos[base_name]
                pos.append([float(coord[0] * 100.0), float(coord[1] * 100.0), float(coord[2] * 100.0)])
        pos_tensor = torch.tensor(pos, dtype=torch.float32)
        if missing and key not in self._missing_pos_warned:
            print(f"Warning: Missing coordinates for channels: {missing}. Using zeros.")
            self._missing_pos_warned.add(key)
        self._pos_cache[key] = pos_tensor
        return pos_tensor

    def _fix_time_length(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_num_samples is None:
            self.fixed_num_samples = int(x.shape[-1])
            return x

        if x.shape[-1] == self.fixed_num_samples:
            return x
        if x.shape[-1] > self.fixed_num_samples:
            return x[..., : self.fixed_num_samples]

        pad_len = self.fixed_num_samples - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad_len), mode="constant", value=0.0)

    def _prepare_batch(self, xb: torch.Tensor, channels) -> Tuple[torch.Tensor, torch.Tensor]:
        if xb.ndim == 2:
            xb = xb.unsqueeze(0)
        xb = xb.to(self.device, dtype=torch.float32)

        ch_names = self._extract_channels(channels)
        target_channels = self.default_channels if self.default_channels else ch_names
        if not target_channels:
            target_channels = [f"CH{i}" for i in range(xb.shape[1])]

        xb = self._align_channels(xb, ch_names, target_channels)
        xb = self._fix_time_length(xb)

        pos = self._build_positions(target_channels).to(self.device)
        pos = pos.unsqueeze(0).expand(xb.shape[0], -1, -1)
        return xb, pos

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        self.task_name = meta[0]["task_name"]
        self.default_channels = get_channels(self.task_name)
        self.fixed_num_samples = None

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

        train_batch_size = 1 if self.chunk_len_s is None else self.batch_size
        train_loader = DataLoader(dataset_train, batch_size=train_batch_size, num_workers=0, shuffle=True)

        output_dim = self.num_classes * (self.num_labels_per_chunk if self.is_multilabel_task else 1)
        with torch.no_grad():
            for xb, _, channels in train_loader:
                xb, posb = self._prepare_batch(xb, channels)
                latents = self.model.encode(xb, posb)
                self.model.init_head(latents, out_dim=output_dim)
                break

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
            for xb, yb, channels in tqdm(train_loader, desc="Training", leave=False):
                xb, posb = self._prepare_batch(xb, channels)
                yb = torch.as_tensor(yb, device=self.device).long()
                xb, y_a, y_b, lam = self._apply_mixup(xb, yb)

                optimizer.zero_grad()
                logits = self.model(xb, posb)
                if self.is_multilabel_task:
                    logits = logits.reshape((logits.shape[0], self.num_classes, -1))
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
        test_batch_size = 1 if self.chunk_len_s is None else self.batch_size
        loader = DataLoader(dataset, batch_size=test_batch_size, num_workers=0, shuffle=False)

        self.model.eval()
        predictions = []
        for xb, _, channels in tqdm(loader, desc="Testing", leave=False):
            xb, posb = self._prepare_batch(xb, channels)
            logits = self.model(xb, posb)
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
