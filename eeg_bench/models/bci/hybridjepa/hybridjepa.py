import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

###################
###  MULTI-GPU  ###
###################


def setup_ddp(rank: int, world_size: int):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


##################
###   CONFIG   ###
##################


class JEPAConfig:
    data_dir: Path = Path("/mnt/nvme/prepared")

    epochs: int = 10
    batch_size: int = 512
    lr: float = 5e-5

    fs: int = 200

    P1: int = 200
    S1: int = 180

    P2: int = 400
    S2: int = 360

    D: int = 512

    encoder_heads: int = 8
    encoder_depth: int = 12

    predictor_heads: int = 8
    predictor_depth: int = 4

    decoder_heads: int = 8
    decoder_depth: int = 4

    mask_ratio: float = 0.5

    w_var: float = 10.0
    w_cov: float = 10.0

    m: float = 0.998

    ckpt_dir: Path = Path("/mnt/nvme/checkpoints/HybridJEPA")
    resume_path: Path = None


#################
### UTILITIES ###
#################


def generate_future_mask(B: int, C: int, N: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    # N = (C * num_patches)
    num_patches = N // C
    t_start = int(num_patches * (1 - mask_ratio))

    mask_grid = torch.zeros(C, num_patches, dtype=bool, device=device)  # (C, num_patches)
    mask_grid[:, t_start:] = True
    mask_flat = mask_grid.flatten(0)  # (N,) or (C * num_patches)

    mask = mask_flat.unsqueeze(0).repeat(B, 1)  # (B, N)

    return mask


# NOTE: Suggestion from AI
def generate_random_mask(B: int, N: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    # N is total tokens (C * num_patches)
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)

    mask = torch.ones([B, N], device=device, dtype=torch.bool)
    mask[:, :len_keep] = False  # These are context

    # Unshuffle to get original order
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask


# def smooth_l1_loss(z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
#     D = z_pred.shape[-1]

#     zp = F.layer_norm(z_pred, (D,))
#     zt = F.layer_norm(z_tgt, (D,))

#     return F.smooth_l1_loss(zp, zt)


# # NOTE: Suggestion from AI
# def var_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
#     """
#     z: (B, *, D)  -> flatten to (N,D)
#     penalize low std dims
#     """
#     zf = z.reshape(-1, z.shape[-1]).float()
#     zf = zf - zf.mean(dim=0, keepdim=True)
#     std = torch.sqrt(zf.var(dim=0, unbiased=False) + eps)
#     return F.relu(1 - std).mean()


# # NOTE: Suggestion from AI
# def cov_loss(z: torch.Tensor) -> torch.Tensor:
#     # 1. FLATTEN: Handle (B, N, D) -> (B*N, D)
#     if z.ndim == 3:
#         z = z.reshape(-1, z.shape[-1])

#     batch_size, num_features = z.shape

#     # 2. CENTER: Subtract mean (centering the batch)
#     z = z - z.mean(dim=0, keepdim=True)

#     # 3. COVARIANCE: Calculate Covariance Matrix (D, D)
#     # Note: We divide by (batch_size - 1) for unbiased estimator
#     cov = (z.T @ z) / (batch_size - 1)

#     # 4. OFF-DIAGONAL: We want to push off-diagonals to 0
#     # This trick extracts off-diagonal elements efficiently
#     off_diag = cov.flatten()[:-1].view(num_features - 1, num_features + 1)[:, 1:].flatten()

#     # 5. LOSS: Sum of squares of off-diagonals
#     loss = off_diag.pow(2).sum() / num_features

#     return loss


def compute_recon_loss(x_recon, x_tgt):
    z_tgt_norm = F.layer_norm(x_tgt, (x_tgt.size(-1),))
    loss = F.smooth_l1_loss(x_recon, z_tgt_norm)
    return loss, {"loss": loss.detach()}


def compute_var_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = z.flatten(0, -2)
    std = z.std(dim=0, unbiased=False) + eps
    return torch.relu(1.0 - std).mean()


def compute_cov_loss(z: torch.Tensor) -> torch.Tensor:
    z = z.flatten(0, -2)
    z = z - z.mean(dim=0, keepdim=True)

    N = z.size(0)
    cov = (z.T @ z) / (N - 1)

    cov_squared = cov.pow(2).sum()
    diag_squared = torch.diagonal(cov).pow(2).sum()

    off_diag_squared = cov_squared - diag_squared
    D = cov.size(0)

    return off_diag_squared / (D * (D - 1))


def compute_vic_loss(z_pred: torch.Tensor, z_tgt: torch.Tensor, w_var: float, w_cov: float):
    # z_pred: (B, N, D);
    loss_l1 = F.smooth_l1_loss(z_pred, z_tgt)

    loss_var = compute_var_loss(z_pred)
    loss_cov = compute_cov_loss(z_pred)

    loss = loss_l1 + (w_var * loss_var) + (w_cov * loss_cov)

    return loss, {"loss": loss.detach(), "l1": loss_l1.detach(), "var": loss_var.detach(), "cov": loss_cov.detach()}


##################
###  DATASET   ###
##################
class EEGDataset(Dataset):
    def __init__(self, data_dir: Path):
        super().__init__()

        self.data_dir = data_dir
        self.samples = sorted([p.parent for p in self.data_dir.rglob("x.npy")])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_path = self.samples[idx]

        x = torch.from_numpy(np.load(sample_path / "x.npy")).float()  # (C, T)
        xyz = torch.from_numpy(np.load(sample_path / "xyz.npy")).float()  # (C, 3)

        return x, xyz


#################
### MODELLING ###
#################


class EEGTokenizer(nn.Module):
    def __init__(self, P: int, S: int, D: int, fs: int):
        # P: patch size (200 Hz / 1s), S: step size (180 Hz / 0.9s) (10% overlap);
        # D: token vector len (dimension) (512 / 768);

        super().__init__()

        self.P = P
        self.S = S
        self.fs = fs

        self.linear = nn.Linear(P, D)

    def create_xyzt(self, xyz: torch.Tensor, num_patches: int) -> torch.Tensor:
        # xyz: (B, C, 3), N: (1 + (P - T) / S);
        B, C, _ = xyz.shape

        xyz_expanded = xyz.unsqueeze(2).expand(B, C, num_patches, 3)  # (B, C, 3) -> (B, C, num_patches, 3)

        t_indices = torch.arange(0, num_patches, device=xyz.device, dtype=xyz.dtype).float()  # (N,)
        t = (t_indices * self.S) / self.fs
        t_expanded = t.view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)  # (N,) -> (B, C, N, 1)

        scale_expanded = torch.full((B, C, num_patches, 1), self.P / self.fs, device=xyz.device, dtype=xyz.dtype).float()

        coords = torch.cat([xyz_expanded, t_expanded, scale_expanded], dim=-1)  # (B, C, num_patches, 5)
        coords = coords.flatten(1, 2)  # (B, N, 5)

        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T);
        x = x.unfold(dimension=-1, size=self.P, step=self.S)  # (B, C, num_patches, P)
        patches = x.flatten(1, 2)

        tokens = self.linear(x)  # (B, C, num_patches, D)
        tokens = tokens.flatten(1, 2)  # (B, N, D)

        return tokens, patches


class PositionalEncoder(nn.Module):
    def __init__(self, D: int):
        super().__init__()

        self.register_buffer("freqs", torch.randn(5, D // 2))
        self.project = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D), nn.LayerNorm(D))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N, 5)
        pre_projection = (coords @ self.freqs) * 2 * math.pi
        features = torch.cat([torch.sin(pre_projection), torch.cos(pre_projection)], dim=-1)
        return self.project(features)  # (B, N, D)


class TransformerBlock(nn.Module):
    def __init__(self, D: int, heads: int):
        super().__init__()

        self.pre_attn_norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=heads, batch_first=True)

        self.post_attn_norm = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Nc, D); Nc = number of context tokens (visible);
        h = self.pre_attn_norm(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        h = self.post_attn_norm(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out

        return x


class Encoder(nn.Module):
    def __init__(self, D: int, heads: int, depth: int):
        super().__init__()

        self.blocks = nn.ModuleList([TransformerBlock(D, heads) for I in range(depth)])
        self.final_norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class Predictor(nn.Module):
    def __init__(self, D: int, heads: int, depth: int):
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.mask_token, std=0.02)

        self.predict = Encoder(D, heads, depth)
        self.project = nn.Linear(D, D)

    def forward(self, z_ctx: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, pos: PositionalEncoder) -> torch.Tensor:
        # ctx: (B, Nc, D), mask: (B, N) (0/1);
        B, N = mask.shape
        _, _, D = z_ctx.shape

        canvas = self.mask_token.expand(B, N, D).clone()  # (B, N, D)

        for b in range(B):
            canvas[b, ~mask[b]] = z_ctx[b]

        pe = pos(coords)  # (B, N, D)
        canvas = canvas + pe * (mask).unsqueeze(-1).type_as(canvas)

        prediction = self.predict(canvas)
        prediction = self.project(prediction)

        return prediction


class Decoder(nn.Module):
    def __init__(self, D: int, P: int, heads: int, depth: int):
        super().__init__()

        self.blocks = nn.ModuleList([TransformerBlock(D, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(D)

        self.head = nn.Linear(D, P)

    def forward(self, z_pred: torch.Tensor, pe_masked: torch.Tensor) -> torch.Tensor:
        # z_pred: (B, Nt, D), pe_masked: (B, Nt, D) -> The positional encodings for just the masked tokens;
        x = z_pred + pe_masked

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        return self.head(x)  # (B, Nt, P)


class JEPA(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()

        self.mask_ratio = config.mask_ratio
        self.w_var = config.w_var
        self.w_cov = config.w_cov

        self.tokenizer_short = EEGTokenizer(config.P1, config.S1, config.D, config.fs)
        # self.tokenizer_long = EEGTokenizer(config.P2, config.S2, config.D, config.fs)

        self.pos = PositionalEncoder(config.D)

        self.student = Encoder(config.D, config.encoder_heads, config.encoder_depth)
        self.teacher = Encoder(config.D, config.encoder_heads, config.encoder_depth)

        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(config.D, config.predictor_heads, config.predictor_depth)

        self.decoder_short = Decoder(config.D, config.P1, config.decoder_heads, config.decoder_depth)
        # self.decoder_long = Decoder(config.D, config.P2, config.decoder_heads, config.decoder_depth)

    @torch.no_grad()
    def update_ema(self, m: float):
        for teacher_params, student_params in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_params.data.mul_(m).add_(student_params.data, alpha=1 - m)

    def encode(self, x: torch.Tensor, xyz: torch.Tensor) -> float:
        B, C, _ = x.shape

        # tokens_s, _ = tokenizer(x_student)
        tokens, _ = self.tokenizer_short(x)

        _, N, D = tokens.shape

        num_patches = N // C
        coords = self.tokenizer_short.create_xyzt(xyz, num_patches)

        pe = self.pos(coords)
        tokens = tokens + pe

        return self.teacher(tokens)


    def _call(self, tokenizer: EEGTokenizer, x_student: torch.Tensor, x_teacher: torch.Tensor, xyz: torch.Tensor) -> float:
        B, C, _ = x_student.shape

        tokens_s, _ = tokenizer(x_student)
        tokens_t, x = tokenizer(x_teacher)

        _, N, D = tokens_s.shape

        num_patches = N // C
        coords = tokenizer.create_xyzt(xyz, num_patches)

        pe = self.pos(coords)

        tokens_s = tokens_s + pe
        tokens_t = tokens_t + pe

        mask = generate_random_mask(B, N, self.mask_ratio, device=tokens_s.device)

        Nc = int((~mask[0]).sum().item())
        Nt = int((mask[0].sum().item()))

        ctx_tokens = tokens_s[~mask].view(B, Nc, D)
        z_ctx = self.student(ctx_tokens)

        with torch.no_grad():
            z_all = self.teacher(tokens_t)
            z_tgt = z_all[mask].view(B, Nt, D)
            z_tgt = z_tgt.detach()

        prediction = self.predictor(z_ctx, coords, mask, self.pos)
        z_pred = prediction[mask].view(B, Nt, D)

        pe_masked = pe[mask].view(B, Nt, D)
        decoder = self.decoder_short if tokenizer.P == self.tokenizer_short.P else self.decoder_long

        x_recon = decoder(z_pred, pe_masked).view(B, Nt, -1)
        x_tgt = x[mask].view(B, Nt, -1)

        loss_recon, stats_recon = compute_recon_loss(x_recon, x_tgt)
        loss_vic, stats_vic = compute_vic_loss(z_pred, z_tgt, self.w_var, self.w_cov)

        loss = loss_vic + loss_recon

        return loss, {"loss": loss.detach(), "loss_vic": loss_vic.detach(), "loss_recon": loss_recon.detach(), "l1": stats_vic["l1"], "var": stats_vic["var"], "cov": stats_vic["cov"]}

    def forward(self, x_student: torch.Tensor, x_teacher: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        loss_s, stats_s = self._call(self.tokenizer_short, x_student, x_teacher, xyz)
        # loss_l, stats_l = self._call(self.tokenizer_long, x_student, x_teacher, xyz)

        loss = loss_s
        return loss, {"loss": loss.detach(), "loss_vic": stats_s["loss_vic"], "loss_recon": stats_s["loss_recon"], "loss_l1": stats_s["l1"], "loss_var": stats_s["var"], "loss_cov": stats_s["cov"]}


class HybridJEPAWithClassifier(nn.Module):
    def __init__(self, num_classes, checkpoint_path, embedding_dim=None, freeze=False):
        super().__init__()
        
        self.jepa = JEPA(JEPAConfig())

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        # Handle both state_dict and full model saves
        if isinstance(checkpoint, dict):
            # It's a state_dict or a checkpoint dict
            if 'model_state_dict' in checkpoint:
                self.jepa.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.jepa.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume it's a raw state_dict
                self.jepa.load_state_dict(checkpoint)
        elif isinstance(checkpoint, nn.Module):
            # It's a full model object
            self.jepa = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

        if freeze:
            for param in self.jepa.parameters():
                param.requires_grad = False

        classifier_input_dim = embedding_dim if embedding_dim is not None else 256
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(classifier_input_dim),
            nn.Dropout(0.1),
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, x, xyz):
        z = self.jepa.encode(x, xyz)
        logits = self.final_layer(z)
        return logits

@torch.no_grad()
def augment_x(x: torch.Tensor):
    # x: (B,C,T);

    if torch.rand(()) < 0.8:
        scale = torch.empty((x.size(0), 1, 1), device=x.device).uniform_(0.7, 1.3)
        x = x * scale

    if torch.rand(()) < 0.5:
        x = x + 0.01 * torch.randn_like(x)

    return x


#####################
### TRAINING LOOP ###
#####################



def train(config: JEPAConfig):
    world_size = torch.cuda.device_count()
    rank = int(os.environ["LOCAL_RANK"])

    setup_ddp(rank, world_size)

    if rank == 0:
        config.ckpt_dir.mkdir(parents=True, exist_ok=True)

    effective_lr = config.lr * world_size

    dataset = EEGDataset(config.data_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(dataset, config.batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    model = JEPA(config).cuda(rank)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=effective_lr, weight_decay=0.05)

    model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)

    progress = 0
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        step = 0

        for x, xyz in data_loader:
            x, xyz = x.cuda(rank), xyz.cuda(rank)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_student = augment_x(x)
                loss, stats = model(x_student, x, xyz)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # EMA Schedule
            m = config.m + (0.9999 - config.m) * min(1.0, progress / 25000)
            model.module.update_ema(m)

            if rank == 0 and step % 100 == 0:
                print(
                    f"--- Epoch {epoch:03d}, Step {step:06d} ---\n\t[ loss {stats['loss'].item():.4f} | loss_vic {stats['loss_vic'].item():.4f} | loss_recon {stats['loss_recon'].item():.4f} | loss_l1 {stats['loss_l1'].item():.4f} | loss_var {stats['loss_var'].item():.4f} | loss_cov {stats['loss_cov'].item():.4f} ]\n"
                )

            step += 1
            progress += 1

        if rank == 0:
            checkpoint_path = config.ckpt_dir / f"epoch_{epoch + 1}.pth"
            torch.save({"model_state_dict": model.module.state_dict(), "optimizer_state_dict": opt.state_dict()}, checkpoint_path)
            print(f"--- Saved checkpoint for epoch {epoch} to {checkpoint_path} ---")

    cleanup()


if __name__ == "__main__":
    train(JEPAConfig())