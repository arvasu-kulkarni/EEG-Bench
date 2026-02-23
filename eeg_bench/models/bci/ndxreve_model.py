from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

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


def _make_channel_positions(channels: int) -> torch.Tensor:
    theta = np.linspace(0.0, 2.0 * math.pi, channels + 1, endpoint=True)[:-1]
    phi = np.linspace(0.2 * math.pi, 0.8 * math.pi, channels, endpoint=True)
    xs = np.sin(phi) * np.cos(theta)
    ys = np.sin(phi) * np.sin(theta)
    zs = np.cos(phi)
    positions = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
    return torch.from_numpy(positions)


class NDXReveDataset(Dataset):
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
        # Fallback to base attention if masks are passed; ndx REVE path uses no masks.
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


class NDXReveClassifier(nn.Module):
    def __init__(self, mae: nn.Module, dropout: float = 0.1, pooling_mode: str = "attention"):
        super().__init__()
        self.patch_embed = mae.patch_embed
        self.pos_enc = mae.pos_enc
        self.encoder = mae.encoder
        self.aux_query = getattr(mae, "aux_query", None)
        self.aux_linear = getattr(mae, "aux_linear", None)
        self.patch_size = int(mae.patch_size)
        self.step = int(mae.step)
        self.dropout = float(dropout)
        self.pooling_mode = str(pooling_mode).lower().strip()
        if self.pooling_mode not in {"attention", "mean", "flatten"}:
            raise ValueError(f"Unsupported pooling_mode={pooling_mode}")
        if self.pooling_mode == "attention" and (self.aux_query is None or self.aux_linear is None):
            print("Warning: attention pooling requested but aux_query/aux_linear missing; falling back to mean pooling.")
            self.pooling_mode = "mean"
        self.final_layer: Optional[nn.Module] = None
        self.feature_dim: Optional[int] = None

    def prepare_coords(self, xyz: torch.Tensor, num_patches: int) -> torch.Tensor:
        bsz, channels, _ = xyz.shape
        device = xyz.device
        time_idx = torch.arange(num_patches, device=device, dtype=torch.float32)
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)
        time = time_idx.view(1, 1, num_patches, 1).expand(bsz, channels, -1, -1)
        return torch.cat([spat, time], dim=-1).flatten(1, 2)

    def encode(self, x: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]
        tokens = self.patch_embed.linear(patches).flatten(1, 2)
        coords = self.prepare_coords(pos, num_patches)
        pe = self.pos_enc(coords)
        latents, intermediates = self.encoder(tokens + pe)
        return latents, intermediates

    def _pool_features(self, latents: torch.Tensor, intermediates: List[torch.Tensor]) -> torch.Tensor:
        if self.pooling_mode == "flatten":
            return latents.flatten(1)
        if self.pooling_mode == "mean":
            return latents.mean(dim=1)

        aux_input = torch.cat(intermediates, dim=-1)
        aux_query = self.aux_query
        aux_linear = self.aux_linear
        if aux_query is None or aux_linear is None:
            return latents.mean(dim=1)

        attn_scores = torch.matmul(aux_input, aux_query.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=1)
        global_token = torch.sum(attn_weights * aux_input, dim=1, keepdim=True)
        global_emb = aux_linear(global_token).squeeze(1)
        return global_emb

    def init_head(self, latents: torch.Tensor, intermediates: List[torch.Tensor], out_dim: int) -> None:
        features = self._pool_features(latents, intermediates)
        in_dim = int(features.shape[-1])
        if self.final_layer is not None and self.feature_dim == in_dim:
            return
        self.feature_dim = in_dim
        self.final_layer = nn.Sequential(
            nn.RMSNorm(in_dim),
            nn.Dropout(self.dropout),
            nn.Linear(in_dim, out_dim),
        ).to(features.device)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        latents, intermediates = self.encode(x, pos)
        if self.final_layer is None:
            raise RuntimeError("Classifier head is not initialized.")
        features = self._pool_features(latents, intermediates)
        return self.final_layer(features)


class NdxReveModel(AbstractModel):
    def __init__(
        self,
        checkpoint_path: str = "/home/neurodx/mahir/ndx-pipeline/runs/sanity_check/mae_epoch_50.pt",
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
        epochs: int = 30,
        lr: float = 3e-5,
        weight_decay: float = 2e-4,
        num_classes: int = 2,
        linear_probe_epochs: int = 3,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.1,
        mixup_alpha: float = 0.4,
        plateau_factor: float = 0.5,
        plateau_patience: int = 3,
        plateau_min_lr: float = 1e-6,
        dropout: float = 0.1,
        pooling_mode: str = "attention",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_only_finetune: bool = True,
    ):
        super().__init__("NdxReveModel")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.num_classes = int(num_classes)
        self.linear_probe_epochs = max(0, int(linear_probe_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(warmup_start_factor)
        self.mixup_alpha = float(mixup_alpha)
        self.plateau_factor = float(plateau_factor)
        self.plateau_patience = int(plateau_patience)
        self.plateau_min_lr = float(plateau_min_lr)
        self.target_fs = int(fs)
        self.pooling_mode = str(pooling_mode).lower().strip()
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_only_finetune = bool(lora_only_finetune)
        self.lora_enabled = False

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

        self.model = NDXReveClassifier(mae=mae, dropout=dropout, pooling_mode=self.pooling_mode).to(self.device)
        self._inject_lora_into_attention()
        self.task_name: Optional[str] = None
        self.task_num_classes: Optional[int] = None

        self._standard_19_coords = self._load_standard_19_coords(ndx_model_py_path)
        self._pos_cache: Dict[Tuple[Optional[str], Optional[str], int], torch.Tensor] = {}

    @staticmethod
    def _load_standard_19_coords(model_py_path: str) -> Optional[torch.Tensor]:
        montage_path = Path(model_py_path).with_name("montage.py")
        if not montage_path.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location("ndx_pipeline_montage", montage_path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, "get_standard_19_coords_numpy", None)
            if fn is None:
                return None
            coords = fn()
            coords_np = np.asarray(coords, dtype=np.float32)
            if coords_np.ndim != 2 or coords_np.shape[1] != 3:
                return None
            return torch.from_numpy(coords_np).float()
        except Exception:
            return None

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
                continue
            is_lora_param = "lora_" in name
            if not train_backbone:
                param.requires_grad = False
            elif self.lora_enabled and self.lora_only_finetune:
                param.requires_grad = is_lora_param
            else:
                param.requires_grad = train_backbone

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

    @staticmethod
    def _coords_tensor_from_any(value: Any, channels: int) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (channels, 3):
            return torch.from_numpy(arr.astype(np.float32, copy=False))
        return None

    @staticmethod
    def _coords_tensor_from_path(path: Optional[str], channels: int) -> Optional[torch.Tensor]:
        if not path or not os.path.exists(path):
            return None
        try:
            arr = np.load(path).astype(np.float32)
        except Exception:
            return None
        if arr.ndim == 2 and arr.shape == (channels, 3):
            return torch.from_numpy(arr)
        return None

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

    def _build_positions(self, meta_item: Dict, channels: int) -> torch.Tensor:
        coords_path = cast(Optional[str], meta_item.get("coords_path")) if isinstance(meta_item, dict) else None
        signal_path = cast(Optional[str], meta_item.get("signal_path")) if isinstance(meta_item, dict) else None
        inline_coords = meta_item.get("coords") if isinstance(meta_item, dict) else None
        key = (coords_path, signal_path, int(channels))
        if inline_coords is None and key in self._pos_cache:
            return self._pos_cache[key]

        # Match ndx-pipeline train.py order: explicit coords -> sibling coords.npy -> std19 -> synthetic.
        pos_tensor = self._coords_tensor_from_any(inline_coords, channels)
        if pos_tensor is None:
            pos_tensor = self._coords_tensor_from_path(coords_path, channels)
        if pos_tensor is None and signal_path:
            sibling_coords = os.path.join(os.path.dirname(signal_path), "coords.npy")
            pos_tensor = self._coords_tensor_from_path(sibling_coords, channels)
        if pos_tensor is None and self._standard_19_coords is not None and channels == int(self._standard_19_coords.shape[0]):
            pos_tensor = self._standard_19_coords.clone()
        if pos_tensor is None:
            pos_tensor = _make_channel_positions(channels)

        if inline_coords is None:
            self._pos_cache[key] = pos_tensor
        return pos_tensor

    def normalize(self, data: np.ndarray) -> np.ndarray:
        data_mean = data.mean(axis=(0, 2), keepdims=True)
        data_std = data.std(axis=(0, 2), keepdims=True) + 1e-6
        return (data - data_mean) / data_std

    def _resample(self, data: np.ndarray, meta: Dict) -> np.ndarray:
        orig_freq = int(meta["sampling_frequency"])
        if orig_freq == self.target_fs:
            return data.astype(np.float32, copy=False)

        print(f"Resampling from {orig_freq} Hz to {self.target_fs} Hz")
        data_float = data.astype(np.float64, copy=False)
        if orig_freq < self.target_fs:
            resampled = mne.filter.resample(data_float, up=self.target_fs / orig_freq, verbose=False)
        else:
            resampled = mne.filter.resample(data_float, down=orig_freq / self.target_fs, verbose=False)
        return resampled.astype(np.float32, copy=False)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        self.task_name = meta[0]["task_name"]
        self.task_num_classes = n_unique_labels(self.task_name)

        class_weights = torch.tensor(calc_class_weights(y, self.task_name), dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        loaders = []
        for x_arr, y_arr, meta_item in zip(cast(List[np.ndarray], X), cast(List[np.ndarray], y), meta):
            if x_arr.ndim != 3 or x_arr.size == 0:
                continue
            data = self._resample(x_arr, meta_item)
            data = self.normalize(data)
            labels = np.array([map_label(label, self.task_name) for label in y_arr], dtype=np.int64)
            pos = self._build_positions(meta_item, channels=int(data.shape[1]))
            dataset = NDXReveDataset(data=data, labels=labels, pos=pos)
            loaders.append(DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0))

        if not loaders:
            return

        with torch.no_grad():
            for loader in loaders:
                for xb, _, posb in loader:
                    xb = xb.to(self.device)
                    posb = posb.to(self.device)
                    latents, intermediates = self.model.encode(xb, posb)
                    self.model.init_head(latents, intermediates, out_dim=self.task_num_classes)
                    break
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
            for loader in loaders:
                for xb, yb, posb in tqdm(loader):
                    xb = xb.to(self.device)
                    yb = yb.to(self.device).long()
                    posb = posb.to(self.device)

                    xb, y_a, y_b, lam = self._apply_mixup(xb, yb)

                    optimizer.zero_grad()
                    logits = self.model(xb, posb)
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
        if self.task_name is None or self.task_num_classes is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        self.model.eval()
        predictions = []
        for x_arr, meta_item in zip(cast(List[np.ndarray], X), meta):
            if x_arr.ndim != 3 or x_arr.size == 0:
                continue
            data = self._resample(x_arr, meta_item)
            data = self.normalize(data)
            pos = self._build_positions(meta_item, channels=int(data.shape[1]))

            dataset = NDXReveDataset(data, None, pos)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            batch_preds = []
            for xb, posb in tqdm(loader):
                xb = xb.to(self.device)
                posb = posb.to(self.device)
                logits = self.model(xb, posb)
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
