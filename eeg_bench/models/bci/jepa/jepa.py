import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# ============================================================
# Distributed init / cleanup
# ============================================================


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def ddp_cleanup():
    dist.destroy_process_group()


# ============================================================
# Dataset
# ============================================================


class EEGDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = sorted([p.parent for p in self.data_dir.rglob("x.npy")])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        part = self.samples[idx]
        # Optimization: use mmap_mode='r' if files are large to avoid RAM spikes,
        # but for small EEG segments load is fine.
        x = torch.from_numpy(np.load(part / "x.npy")).float()  # [19, 2000]
        xyz = torch.from_numpy(np.load(part / "xyz.npy")).float()  # [19, 3]
        return x, xyz


# ============================================================
# Vectorized Masking
# ============================================================


def sample_block_mask(B, C, P, device):
    """
    Generates a boolean mask [B, C*P] where True indicates MASKED (hidden).
    Vectorized to ensure different masks per sample in the batch.
    """
    # 1. Temporal Blocks
    # Create a grid of time indices [1, 1, P]
    time_indices = torch.arange(P, device=device).view(1, 1, P)

    # Random lengths per sample: [B, 1, 1]
    t_lens = torch.randint(int(0.2 * P), int(0.4 * P) + 1, (B, 1, 1), device=device)

    # Random start indices per sample: [B, 1, 1]
    # We must ensure start + len <= P
    t_starts = (torch.rand((B, 1, 1), device=device) * (P - t_lens)).long()

    # Mask logic: start <= index < start + len
    t_mask = (time_indices >= t_starts) & (time_indices < (t_starts + t_lens))  # [B, 1, P]

    # 2. Channel Blocks
    # We want to select K channels to mask entirely
    # There is no easy vectorized way to do "random subset without replacement" perfectly
    # without a loop or argsort. Randperm via argsort is best.

    # Random noise for sorting [B, C]
    rand_c = torch.rand(B, C, device=device)
    # Get indices of channels to mask
    _, c_indices = torch.sort(rand_c, dim=1)

    # Determine how many channels to mask per sample [B, 1]
    n_mask_ch = torch.randint(3, min(7, C + 1), (B, 1), device=device)

    # Create channel mask matrix [B, C]
    # We need to see if the rank (0..C-1) is less than n_mask_ch
    c_rank = torch.argsort(c_indices, dim=1)  # The rank of each channel
    c_mask = c_rank < n_mask_ch  # [B, C]

    # 3. Combine
    # Expand dims to broadcast: [B, C, 1] | [B, 1, P]
    # We explicitly want the union of channel masking AND time masking?
    # The original code did: mask time block everywhere, AND mask specific channels everywhere.
    # Usually, JEPA uses BLOCK masking (random rectangular blocks).
    # We will replicate the original logic: Union of time-strip + channel-strip.

    full_mask = t_mask | c_mask.unsqueeze(-1)  # [B, C, P]

    return full_mask.view(B, C * P)


# ============================================================
# Tokenizer + Positional
# ============================================================


class EEGTokenizer(nn.Module):
    # CHANGE: patch_size=50 (was 200)
    def __init__(self, channels=19, patch_size=200, feat_per_channel=8, embed_dim=256):
        super().__init__()
        self.C = channels
        self.PS = patch_size
        self.F = feat_per_channel
        self.D = embed_dim

        # Conv1d as Tokenizer
        self.conv = nn.Conv1d(channels, channels * feat_per_channel, kernel_size=patch_size, stride=patch_size, groups=channels)
        self.proj = nn.Linear(feat_per_channel, embed_dim)

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape

        # Tokenize
        y = self.conv(x)  # [B, C*F, P]
        _, _, P = y.shape

        # Prepare Targets for MAE (raw pixels)
        patches = x.view(B, C, P, self.PS)
        patch_targets = patches.view(B, C * P, self.PS)

        # Reshape to [B, C, P, F] -> [B, C*P, F] -> Proj -> [B, C*P, D]
        y = y.view(B, C, self.F, P).permute(0, 1, 3, 2)
        tokens = self.proj(y).view(B, C * P, self.D)

        return tokens, patch_targets, P


class XYZTPositional(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.xyz_mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, embed_dim))
        self.time_emb = nn.Embedding(512, embed_dim)

    def forward(self, xyz, P, device):
        # xyz: [C, 3] or [B, C, 3]. Code passed [19,3].
        if xyz.dim() == 3:
            # If xyz is batch-specific [B, C, 3], we take average or just index 0
            # if we assume standard montage. Let's assume input is [C, 3].
            xyz = xyz[0]

        C = xyz.size(0)
        ch_pos = self.xyz_mlp(xyz)  # [C, D]
        t_pos = self.time_emb(torch.arange(P, device=device))  # [P, D]

        # Construct full positional grid [1, C*P, D]
        # We broadcast to allow efficient addition later
        pos_emb = torch.zeros(C * P, ch_pos.shape[-1], device=device)
        for c in range(C):
            # Add channel embedding to specific chunk
            # Add time embedding to all chunks
            # Better vectorized:
            pos_emb[c * P : (c + 1) * P] = ch_pos[c].unsqueeze(0) + t_pos

        return pos_emb.unsqueeze(0)  # [1, C*P, D]


# ============================================================
# Transformers
# ============================================================


def make_encoder(D, depth, heads, dropout=0.0):
    # Added dropout parameter
    layer = nn.TransformerEncoderLayer(d_model=D, nhead=heads, dim_feedforward=4 * D, batch_first=True, activation="gelu", norm_first=True, dropout=dropout)
    return nn.Sequential(nn.TransformerEncoder(layer, depth), nn.LayerNorm(D))


class Predictor(nn.Module):
    def __init__(self, D, depth, heads):
        super().__init__()
        # Predictor Mask Token: used to query the predictor
        self.mask_token = nn.Parameter(torch.zeros(D))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.enc = make_encoder(D, depth, heads)

    def forward(self, H, mask):
        # H: Student output [B, L, D] (already contains context info)
        # We replace H at mask locations with the Predictor's mask token
        # This tells the predictor: "Please predict what should be here"
        X = H.clone()
        X[mask] = self.mask_token
        X = self.enc(X)
        return X[mask]


class MAEDecoder(nn.Module):
    def __init__(self, D, depth, heads, patch_size):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(D))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.dec = make_encoder(D, depth, heads)
        self.out = nn.Linear(D, patch_size)

    def forward(self, H, mask):
        # Similar to Predictor, prepare input for decoder
        X = H.clone()
        X[mask] = self.mask_token
        X = self.dec(X)
        return self.out(X[mask])


# ============================================================
# Full Model
# ============================================================

class EEG_JEPA_MAE_Classifier(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None):
        super().__init__()
        
        # Always instantiate the model first
        self.jepa_mae = EEG_JEPA_MAE()
        
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            # Handle both state_dict and full model saves
            if isinstance(checkpoint, dict):
                # It's a state_dict or a checkpoint dict
                if 'model_state_dict' in checkpoint:
                    self.jepa_mae.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.jepa_mae.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume it's a raw state_dict
                    self.jepa_mae.load_state_dict(checkpoint)
            elif isinstance(checkpoint, nn.Module):
                # It's a full model object
                self.jepa_mae = checkpoint
            else:
                raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
        
        self.final_layer = nn.Linear(256, num_classes)

    def forward(self, x, xyz):
        H = self.jepa_mae.encode(x, xyz)  # Get student representation
        # Pooling: Mean over tokens
        H_pooled = H.mean(dim=1)
        logits = self.final_layer(H_pooled)
        return logits

class EEG_JEPA_MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = 256
        # CHANGE: Explicit patch_size=50
        self.tokenizer = EEGTokenizer(embed_dim=self.D, patch_size=200)
        self.pos = XYZTPositional(self.D)

        # Learnable Mask Token for the Student Input
        self.student_mask_token = nn.Parameter(torch.zeros(self.D))
        nn.init.trunc_normal_(self.student_mask_token, std=0.02)

        self.student = make_encoder(self.D, depth=12, heads=8)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Predictor (JEPA)
        self.predictor = Predictor(self.D, depth=2, heads=8)

        # Decoder (MAE)
        # CHANGE: patch_size=50 (Must match tokenizer)
        self.mae = MAEDecoder(self.D, depth=4, heads=8, patch_size=200)

    @torch.no_grad()
    def update_teacher(self, m=0.998):
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)

    def encode(self, x, xyz):
        B, C, T = x.shape

        # --- FIX 1: Safety Instance Normalization ---
        # Ensures input to the network is strictly N(0,1) regardless of preprocessing artifacts
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x = (x - x_mean) / (torch.sqrt(x_var) + 1e-6)

        # 1. Tokenize & Positional Embeddings
        tokens, patch_targets, P = self.tokenizer(x)
        # Get raw positional embeddings [1, C*P, D]
        pos_emb = self.pos(xyz, P, x.device)

        # Add position to tokens (Teacher Input)
        tokens_pos = tokens + pos_emb

        # 2. Masking
        mask = sample_block_mask(B, C, P, x.device)

        # 3. Student Input Construction
        student_in = tokens_pos.clone()
        # Broadcast mask token and add positional embedding at masked locations
        mask_token_with_pos = self.student_mask_token + pos_emb
        batch_mask_token_with_pos = mask_token_with_pos.expand(B, -1, -1)
        student_in[mask] = batch_mask_token_with_pos[mask]

        # 4. Forward Passes
        H = self.student(student_in)

        return H

    def forward(self, x, xyz):
        B, C, T = x.shape

        # --- FIX 1: Safety Instance Normalization ---
        # Ensures input to the network is strictly N(0,1) regardless of preprocessing artifacts
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x = (x - x_mean) / (torch.sqrt(x_var) + 1e-6)

        # 1. Tokenize & Positional Embeddings
        tokens, patch_targets, P = self.tokenizer(x)
        # Get raw positional embeddings [1, C*P, D]
        pos_emb = self.pos(xyz, P, x.device)

        # Add position to tokens (Teacher Input)
        tokens_pos = tokens + pos_emb

        # 2. Masking
        mask = sample_block_mask(B, C, P, x.device)

        # 3. Student Input Construction
        student_in = tokens_pos.clone()
        # Broadcast mask token and add positional embedding at masked locations
        mask_token_with_pos = self.student_mask_token + pos_emb
        batch_mask_token_with_pos = mask_token_with_pos.expand(B, -1, -1)
        student_in[mask] = batch_mask_token_with_pos[mask]

        # 4. Forward Passes
        H = self.student(student_in)

        with torch.no_grad():
            Y = self.teacher(tokens_pos)

        # 5. Losses

        # JEPA Loss: Predict Teacher Representation at masked locations
        Z = self.predictor(H, mask)
        Yt = Y[mask]
        jepa_loss = 1 - F.cosine_similarity(F.normalize(Z, dim=-1), F.normalize(Yt, dim=-1), dim=-1).mean()

        # MAE Loss: Predict Raw Pixels at masked locations
        P_pred = self.mae(H, mask)
        P_true = patch_targets[mask]

        # --- FIX 2: Target Normalization (Patch Norm) ---
        # Normalize the target patches. This is MANDATORY for MAE stability.
        p_mean = P_true.mean(dim=-1, keepdim=True)
        p_var = P_true.var(dim=-1, keepdim=True)
        P_true_norm = (P_true - p_mean) / (torch.sqrt(p_var) + 1e-6)

        mae_loss = F.mse_loss(P_pred, P_true_norm)

        loss = jepa_loss


# ============================================================
# Training Loop (DDP)
# ============================================================

import math


def get_lr_schedule(step, max_steps, base_lr, min_lr, warmup_steps):
    """
    Linear Warmup -> Cosine Decay to min_lr
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / max(1, warmup_steps))
    else:
        # Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_momentum_schedule(step, max_steps, start_m=0.996, end_m=1.0):
    """
    Cosine schedule for EMA momentum.
    Starts at start_m, ramps up to end_m (usually 1.0) by the end of training.
    """
    # Standard approach: Cosine increasing from start_m to end_m
    progress = step / max_steps
    return end_m - (end_m - start_m) * 0.5 * (1 + math.cos(math.pi * progress))


def train():
    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank)

    # --- Setup Data ---
    dataset_path = "/mnt/nvme/prepared"
    if not os.path.exists(dataset_path):
        if local_rank == 0:
            print(f"Error: {dataset_path} not found.")
        ddp_cleanup()
        return

    dataset = EEGDataset(dataset_path)
    sampler = DistributedSampler(dataset, shuffle=True)

    # Batch size logic
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)

    # --- Setup Model ---
    model = EEG_JEPA_MAE().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # --- Hyperparameters & Schedulers ---
    epochs = 20
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch

    # LR Config (I-JEPA / MAE standard)
    base_lr = 1.5e-4 * (batch_size * dist.get_world_size() / 256)  # Scale LR by global batch size
    min_lr = 1e-6
    warmup_steps = 2 * steps_per_epoch  # Warmup for 2 epochs (standard is ~10-15 epochs for ImageNet, fewer for smaller data)

    # EMA Config
    ema_start = 0.996
    ema_end = 1.0

    # Optimizer (Weight decay is critical for SSL)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05, betas=(0.9, 0.95))

    if local_rank == 0:
        print(f"Training for {epochs} epochs ({total_steps} steps).")
        print(f"LR: {base_lr:.2e} -> {min_lr:.2e} (Warmup: {warmup_steps} steps)")
        print(f"EMA: {ema_start} -> {ema_end}")

    # --- Loop ---
    global_step = 0

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()

        accum_loss = 0.0

        for i, (x, xyz) in enumerate(loader):
            # 1. Update Hyperparams (Per Step)
            cur_lr = get_lr_schedule(global_step, total_steps, base_lr, min_lr, warmup_steps)
            cur_m = get_momentum_schedule(global_step, total_steps, ema_start, ema_end)

            # Apply LR
            for param_group in opt.param_groups:
                param_group["lr"] = cur_lr

            # 2. Data Move
            x = x.to(device, non_blocking=True)
            xyz = xyz[0].to(device)

            # 3. Forward & Backward
            loss, lj, lm = model(x, xyz)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping is good for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # 4. Update Teacher with Dynamic Momentum
            # Note: We pass the dynamic 'cur_m' here
            model.module.update_teacher(m=cur_m)

            # 5. Logging
            accum_loss += loss.item()
            if local_rank == 0 and i % 50 == 0:
                print(f"[Ep {epoch} | St {global_step}] Loss={loss.item():.4f} (JEPA={lj:.3f} MAE={lm:.3f}) | LR={cur_lr:.2e} | M={cur_m:.5f}")

            global_step += 1

        # --- End of Epoch ---
        if local_rank == 0:
            avg_loss = accum_loss / max(1, steps_per_epoch)
            os.makedirs("/mnt/nvme/checkpoints/MAE-JEPA", exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.module.state_dict(), "opt_state_dict": opt.state_dict(), "scaler": None}, f"/mnt/nvme/checkpoints/MAE-JEPA/epoch_{epoch}.ckpt")  # If using AMP later
            print(f"--- Epoch {epoch} Done. Avg Loss: {avg_loss:.4f} ---")

    ddp_cleanup()


if __name__ == "__main__":
    # To run: torchrun --nproc_per_node=GPU_COUNT script.py
    train()