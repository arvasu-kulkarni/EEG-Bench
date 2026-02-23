from __future__ import annotations

from eeg_bench.models.bci.revefiles.mat import MyReveClassifier


from typing import Dict, List, Optional, Tuple, cast

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.utils_2 import calc_class_weights, map_label, n_unique_labels, reverse_map_label

ignore_chans = ['TPP9h', 'TPP10h', 'AFF1', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'I2', 'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h']

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


class LoRAMultiheadAttention(nn.Module):
    def __init__(self, base_attn: nn.MultiheadAttention, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        if not base_attn.batch_first:
            raise ValueError("Only batch_first MultiheadAttention is supported for LoRA wrapper.")

        self.base = base_attn
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        embed_dim = int(self.base.embed_dim)
        base_weight = self.base.in_proj_weight
        if base_weight is None:
            raise RuntimeError("Expected packed in_proj_weight in MultiheadAttention.")
        base_device = base_weight.device
        base_dtype = base_weight.dtype

        self.lora_A_q = nn.Parameter(torch.empty(self.rank, embed_dim, device=base_device, dtype=base_dtype))
        self.lora_A_k = nn.Parameter(torch.empty(self.rank, embed_dim, device=base_device, dtype=base_dtype))
        self.lora_A_v = nn.Parameter(torch.empty(self.rank, embed_dim, device=base_device, dtype=base_dtype))
        self.lora_A_o = nn.Parameter(torch.empty(self.rank, embed_dim, device=base_device, dtype=base_dtype))

        self.lora_B_q = nn.Parameter(torch.zeros(embed_dim, self.rank, device=base_device, dtype=base_dtype))
        self.lora_B_k = nn.Parameter(torch.zeros(embed_dim, self.rank, device=base_device, dtype=base_dtype))
        self.lora_B_v = nn.Parameter(torch.zeros(embed_dim, self.rank, device=base_device, dtype=base_dtype))
        self.lora_B_o = nn.Parameter(torch.zeros(embed_dim, self.rank, device=base_device, dtype=base_dtype))

        nn.init.kaiming_uniform_(self.lora_A_q, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_A_k, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_A_v, a=np.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_A_o, a=np.sqrt(5.0))

    def _lora_proj(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(self.dropout(x), a), b) * self.scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key_padding_mask is not None or attn_mask is not None:
            return self.base(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        if self.base.in_proj_weight is None:
            raise RuntimeError("Expected packed in_proj_weight in MultiheadAttention.")

        wq, wk, wv = self.base.in_proj_weight.chunk(3, dim=0)
        if self.base.in_proj_bias is not None:
            bq, bk, bv = self.base.in_proj_bias.chunk(3, dim=0)
        else:
            bq = bk = bv = None

        q = F.linear(query, wq, bq) + self._lora_proj(query, self.lora_A_q, self.lora_B_q)
        k = F.linear(key, wk, bk) + self._lora_proj(key, self.lora_A_k, self.lora_B_k)
        v = F.linear(value, wv, bv) + self._lora_proj(value, self.lora_A_v, self.lora_B_v)

        bsz, q_len, _ = q.shape
        k_len = k.shape[1]
        heads = int(self.base.num_heads)
        head_dim = int(self.base.head_dim)

        q = q.view(bsz, q_len, heads, head_dim).transpose(1, 2)
        k = k.view(bsz, k_len, heads, head_dim).transpose(1, 2)
        v = v.view(bsz, k_len, heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=float(self.base.dropout) if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, heads * head_dim)

        out = F.linear(attn_out, self.base.out_proj.weight, self.base.out_proj.bias)
        out = out + self._lora_proj(attn_out, self.lora_A_o, self.lora_B_o)
        return out, None


class EEGEmbedModel(AbstractModel):
    def __init__(
        self,
        checkpoint_path: str = "/home/neurodx/arvasu/EEG-Bench/eeg_bench/models/manas1.pt",
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 2e-4,
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
        dropout: float = 0.1,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_only_finetune: bool = True,
    ):
        super().__init__("EEGEmbedModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.linear_probe_epochs = max(0, int(linear_probe_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(warmup_start_factor)
        self.mixup_alpha = float(mixup_alpha)
        self.plateau_factor = float(plateau_factor)
        self.plateau_patience = int(plateau_patience)
        self.plateau_min_lr = float(plateau_min_lr)
        self.dropout = float(dropout)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_only_finetune = bool(lora_only_finetune)
        self.lora_enabled = False

        dim = embedding_dim if embedding_dim is not None else 45056
        self.model = MyReveClassifier(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            flat_dim=dim,
            dropout=self.dropout,
        ).to(self.device)
        self._inject_lora_into_attention()

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
            elif self.lora_enabled:
                if self.lora_only_finetune:
                    param.requires_grad = is_lora_param
                else:
                    param.requires_grad = True
            else:
                # Requested behavior: without LoRA, stage-2 keeps only final_layer trainable.
                param.requires_grad = False

    @staticmethod
    def _set_module_by_name(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
        parts = module_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _inject_lora_into_attention(self) -> None:
        if self.lora_rank <= 0:
            self.lora_enabled = False
            return

        attention_layers = 0
        for module_name, module in list(self.model.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                wrapped = LoRAMultiheadAttention(
                    module,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                )
                self._set_module_by_name(self.model, module_name, wrapped)
                attention_layers += 1

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

    def _make_positions(self, meta: Dict) -> torch.Tensor:
        # get position from montage assuming 1020
        ch_names = meta["channel_names"]

        # remove ignored channels
        ch_names = [ch for ch in ch_names if ch not in ignore_chans]

        mne_raw_info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        raw_obj = mne.io.RawArray(np.zeros((len(ch_names), 100)), mne_raw_info)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_obj.set_montage(montage, match_case=False, on_missing="raise")
        pos = torch.tensor(list(raw_obj.get_montage().get_positions()["ch_pos"].values()), dtype=torch.float32)
        return 100 * pos

    def _select_channels_and_pos(self, data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, torch.Tensor]:
        selected_data = data.astype(np.float32)
        pos = self._make_positions({"channel_names": ch_names})

        keep_idxs = [i for i, ch in enumerate(ch_names) if ch not in ignore_chans]
        selected_data = selected_data[:, keep_idxs, :]

        return selected_data, pos

    def normalize(self, data: np.ndarray) -> np.ndarray:
        # assuming b x c x t
        data_mean = data.mean(axis=(0, 2), keepdims=True)
        data_std = data.std(axis=(0, 2), keepdims=True) + 1e-6
        normalized = (data - data_mean) / data_std
        return normalized

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

        loaders = []
        for X_, y_, meta_ in zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta):
            if X_.ndim != 3 or X_.size == 0:
                continue
            data, pos = self._select_channels_and_pos(X_, meta_["channel_names"])
            data = self._resample(data, meta_)
            data = self.normalize(data)
            labels = np.array([map_label(label, self.task_name) for label in y_], dtype=np.int64)
            dataset = EEGEmbedBaseDataset(data, labels, pos)
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
