import torch
import torch.nn as nn
import numpy as np
# from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, average_precision_score

# --- IMPORT YOUR MODEL ---
# Ensure model.py is in the directory
from eeg_bench.models.bci.revefiles.model import MAE

# --- CONFIGURATION ---
CHECKPOINT_PATH = "/share/sv7577-h200-41/checkpoints/mae_epoch_50.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 20

# EEGMAT Channels (20 channels)
EEGMAT_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "A2"]

# Coordinates for EEGMAT (x, y, z)
# We manually define these to match the tutorial's geometry
POSITIONS = {
    "Fp1": (-3.09, 11.46, 2.79),
    "Fp2": (2.84, 11.53, 2.77),
    "F3": (-5.18, 8.67, 7.87),
    "F4": (5.03, 8.74, 7.73),
    "F7": (-7.19, 7.31, 2.58),
    "F8": (7.14, 7.45, 2.51),
    "T3": (-8.60, 1.49, 3.12),
    "T4": (8.33, 1.53, 3.10),
    "C3": (-6.71, 2.34, 10.45),
    "C4": (6.53, 2.36, 10.37),
    "T5": (-8.77, 1.29, -0.77),
    "T6": (8.37, 1.17, -0.77),
    "P3": (-5.50, -4.42, 9.99),
    "P4": (5.36, -4.43, 10.05),
    "O1": (-3.16, -8.06, 5.48),
    "O2": (2.77, -8.05, 5.47),
    "Fz": (-0.12, 9.33, 10.26),
    "Cz": (-0.14, 2.76, 14.02),
    "Pz": (-0.17, -4.52, 12.67),
    "A2": (8.39, 0.20, -2.69),
}


# ==========================================
# 1. MODEL WRAPPER (Your MAE -> Classifier)
# ==========================================
class MyReveClassifier(nn.Module):
    def __init__(self, checkpoint_path, num_classes=2, flat_dim=512):
        super().__init__()

        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Initialize YOUR architecture
        self.mae = MAE(fs=200, embed_dim=512, encoder_depth=12, encoder_heads=8, decoder_depth=4, decoder_heads=8, mask_ratio=0.55)
        self.mae.load_state_dict(ckpt["model_state_dict"])

        # Extract necessary components
        self.patch_embed = self.mae.patch_embed
        self.pos_enc = self.mae.pos_enc
        self.encoder = self.mae.encoder
        self.patch_size = self.mae.patch_size
        self.step = self.mae.step

        # Calculate Flatten Dimension
        # 20 channels, 5 seconds.
        # Patch size 1s, stride 0.9s -> 1s window + 4 * 0.9s = 4.6s...
        # Tutorial says output is [B, 20, 5, 512].
        # Let's trust the tutorial dim: 20 * 5 * 512 = 51200

        # 10s segments gives 512x19x11
        self.flat_dim = flat_dim

        # The Head (Same as Tutorial)
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(self.flat_dim),  # Tutorial uses RMSNorm
            nn.Dropout(0.1),
            nn.Linear(self.flat_dim, num_classes),
        )

    def prepare_coords(self, xyz, num_patches):
        # Your logic to expand coords to time
        B, C, _ = xyz.shape
        device = xyz.device
        time_idx = torch.arange(num_patches, device=device).float()
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)
        time = time_idx.view(1, 1, num_patches, 1).expand(B, C, -1, -1)
        return torch.cat([spat, time], dim=-1).flatten(1, 2)

    def forward(self, x, pos):
        # x: (B, 20, 1000) -> 5s @ 200Hz
        # pos: (B, 20, 3)

        # 1. Patchify
        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]  # Should be 5

        # 2. Embed
        tokens = self.patch_embed.linear(patches).flatten(1, 2)

        # 3. Add PE
        coords = self.prepare_coords(pos, num_patches)
        pe = self.pos_enc(coords)

        # 4. Encode
        x_enc = tokens + pe
        latents, _ = self.encoder(x_enc)  # (B, 100, 512)

        # 5. Classify
        return self.final_layer(latents)


# ==========================================
# 2. DATA LOADING (Hugging Face)
# ==========================================
def data():
    print("Loading brain-bzh/eegmat-prepro...")
    dataset = load_dataset("brain-bzh/eegmat-prepro")
    dataset.set_format("torch", columns=["data", "labels"])

    # Create static position tensor
    pos_tensor = torch.tensor([POSITIONS[ch] for ch in EEGMAT_CHANNELS], dtype=torch.float32)


def collate_fn(batch):
    x_data = torch.stack([x["data"] for x in batch])
    y_label = torch.tensor([x["labels"] for x in batch])

    # Repeat coords for batch
    batch_pos = pos_tensor.repeat(len(batch), 1, 1)

    return {"sample": x_data, "label": y_label.long(), "pos": batch_pos}

def loaders():
    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset["val"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==========================================
# 3. SETUP
# ==========================================
def setup():
    model = MyReveClassifier(CHECKPOINT_PATH).to(DEVICE)

    # Freeze Backbone (Tutorial Logic)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_layer.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.final_layer.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


# ==========================================
# 4. TRAINING LOOPS (Tutorial Copy)
# ==========================================
def train_one_epoch(model, optimizer, loader):
    model.train()  # Set to train mode (Dropout active)
    # Ensure backbone stays frozen/eval if needed (Optional depending on model logic)
    model.encoder.eval()

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        data = batch["sample"].to(DEVICE)
        target = batch["label"].to(DEVICE)
        pos = batch["pos"].to(DEVICE)

        optimizer.zero_grad()
        output = model(data, pos)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})


def eval_model(model, loader):
    model.eval()
    y_decisions, y_targets, y_probs = [], [], []
    score, count = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            data = batch["sample"].to(DEVICE)
            target = batch["label"].to(DEVICE)
            pos = batch["pos"].to(DEVICE)

            output = model(data, pos)

            decisions = torch.argmax(output, dim=1)
            score += (decisions == target).int().sum().item()
            count += target.shape[0]

            y_decisions.append(decisions.cpu())
            y_targets.append(target.cpu())
            y_probs.append(output.softmax(dim=1)[:, 1].cpu())  # Prob of class 1

    gt = torch.cat(y_targets).numpy()
    pr = torch.cat(y_decisions).numpy()
    pr_probs = torch.cat(y_probs).numpy()

    return {
        "acc": score / count,
        "balanced_acc": balanced_accuracy_score(gt, pr),
        "kappa": cohen_kappa_score(gt, pr),
        "f1": f1_score(gt, pr, average="weighted"),
        "auroc": roc_auc_score(gt, pr_probs),
        "auc_pr": average_precision_score(gt, pr_probs),
    }

if __name__ == "__main__":
    # ==========================================
    # 5. EXECUTION
    # ==========================================
    print("\n--- Starting Tutorial with Custom Model ---")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(N_EPOCHS):
        train_one_epoch(model, optimizer, train_loader)
        metrics = eval_model(model, val_loader)

        b_acc = metrics["balanced_acc"]
        print(f"Epoch {epoch+1} | Val Bal. Acc: {b_acc:.4f} | F1: {metrics['f1']:.4f}")

        if b_acc > best_val_acc:
            best_val_acc = b_acc
            best_state = model.final_layer.state_dict()

        scheduler.step(b_acc)

    print("\n--- Final Test ---")
    model.final_layer.load_state_dict(best_state)
    results = eval_model(model, test_loader)

    print("-" * 30)
    print(f"Accuracy:      {results['acc']:.4f}")
    print(f"Balanced Acc:  {results['balanced_acc']:.4f}")
    print(f"Cohen Kappa:   {results['kappa']:.4f}")
    print(f"F1 Weighted:   {results['f1']:.4f}")
    print(f"AUROC:         {results['auroc']:.4f}")
    print(f"AUC PR:        {results['auc_pr']:.4f}")
    print("-" * 30)
