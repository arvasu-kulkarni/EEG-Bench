import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, fs: int = 200, patch_seconds: float = 1.0, overlap_seconds: float = 0.1, embed_dim: int = 512):
        super().__init__()

        self.patch_size = int(round(patch_seconds * fs))
        self.overlap_size = int(round(overlap_seconds * fs))

        self.step = self.patch_size - self.overlap_size

        self.linear = nn.Linear(self.patch_size, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.step)
        return self.linear(patches)


class PosEnc(nn.Module):
    def __init__(self, n_freqs: int = 4, embed_dim: int = 512):
        super().__init__()

        freqs = torch.linspace(1.0, 10.0, n_freqs)
        self.register_buffer("freq_matrix", torch.cartesian_prod(freqs, freqs, freqs, freqs).transpose(1, 0))

        fourier_features_dim = 2 * (n_freqs**4)

        self.fourier_linear = nn.Linear(fourier_features_dim, embed_dim, bias=False)
        self.learned_linear = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, coords: torch.Tensor):
        phases = torch.matmul(coords, self.freq_matrix)

        fourier_features = torch.cat([torch.sin(phases), torch.cos(phases)], -1)
        fourier_emb = self.fourier_linear(fourier_features)

        learned_emb = self.learned_linear(coords)

        return self.final_norm(fourier_emb + learned_emb)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()

        assert embed_dim % heads == 0, "dim must be divisible by heads"

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout, batch_first=True)

        self.pre_ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.pre_attn_norm(x)

        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out

        ffn_in = self.pre_ffn_norm(x)

        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out

        return x, ffn_out


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim: int = 512, depth: int = 16, heads: int = 8):
        super().__init__()

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        intermediate = []

        for layer in self.layers:
            x, ffn_out = layer(x)
            intermediate.append(ffn_out)

        return self.final_norm(x), intermediate


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int = 512, decoder_depth: int = 4, decoder_heads: int = 8, patch_size: int = 200):
        super().__init__()

        # 1. The Mask Token (The "Gray Tile")
        # A learnable vector that replaces every missing patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # 2. The Decoder Transformer (Reuse your Encoder logic)
        # It's lighter (fewer layers) than the main Encoder
        self.decoder = TransformerEncoderDecoder(embed_dim=embed_dim, depth=decoder_depth, heads=decoder_heads)

        # 3. The Prediction Head
        # Projects Vector (512) -> Raw Signal (200)
        self.predict = nn.Linear(embed_dim, patch_size, bias=True)

    def forward(self, x_visible: torch.Tensor, pos_enc: nn.Module, coords: torch.Tensor, mask: torch.Tensor):
        B, N_Total, D = coords.shape[0], coords.shape[1], x_visible.shape[-1]

        # --- Step A: Fill Canvas with Mask Tokens ---
        # Create a tensor of size (Batch, Total, Dim) filled with the mask token
        x_full = self.mask_token.expand(B, N_Total, D).clone()

        # --- Step B: Paste Visible Tokens ---
        # Overwrite the mask tokens with the actual encoder output at the visible spots
        for i in range(B):
            # We use the boolean mask to select the "True" slots
            x_full[i, mask[i]] = x_visible[i]

        # --- Step C: Add Positional Encoding ---
        # We call YOUR PosEnc class here.
        # It takes coords (B, N_Total, 4) and returns (B, N_Total, Dim)
        pos_emb = pos_enc(coords)

        # Add GPS info to the tokens
        x_full = x_full + pos_emb

        # --- Step D: Decode ---
        # Pass through the Transformer
        # We ignore the intermediate outputs (the second return value) for now
        x_decoded, _ = self.decoder(x_full)

        # --- Step E: Predict ---
        # (Batch, N_Total, 512) -> (Batch, N_Total, 200)
        prediction = self.predict(x_decoded)

        return prediction


def generate_mask(coords: torch.Tensor, mask_ratio: float = 0.55, spatial_radius: float = 3.0, temporal_radius: float = 3.0):
    B, N, _ = coords.shape
    device = coords.device

    # Calculate exact number of tokens to hide
    num_masked_target = int(mask_ratio * N)

    # Start with all True (Visible)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    for b in range(B):
        spatial_coords = coords[b, :, :3]
        temporal_coords = coords[b, :, 3]

        # --- Phase 1: Block Masking Strategy ---
        # Keep masking blocks until we meet or exceed the target
        while (~mask[b]).sum() < num_masked_target:
            # Pick random seed
            seed_idx = torch.randint(0, N, (1,)).item()

            # Calculate distances
            seed_spatial = spatial_coords[seed_idx]
            dists_spatial = torch.norm(spatial_coords - seed_spatial, dim=1)

            seed_temporal = temporal_coords[seed_idx]
            dists_temporal = torch.abs(temporal_coords - seed_temporal)

            # Find block
            in_block = (dists_spatial <= spatial_radius) & (dists_temporal <= temporal_radius)

            # Mask this block (Set to False)
            mask[b, in_block] = False

        # --- Phase 2: Exact Count Enforcement ---
        # We likely masked too many tokens. We must unmask the excess.

        # Get indices of all tokens that are currently masked
        masked_indices = torch.where(mask[b] == False)[0]
        num_current_masked = len(masked_indices)

        if num_current_masked > num_masked_target:
            # We have excess. Randomly choose which ones to KEEP masked.
            # Shuffle the masked indices
            shuffled_indices = masked_indices[torch.randperm(num_current_masked)]

            # The first 'num_masked_target' stay masked.
            # The rest (excess) must be turned back to Visible (True).
            excess_indices = shuffled_indices[num_masked_target:]

            mask[b, excess_indices] = True

    return mask


class MAE(nn.Module):
    def __init__(
        self,
        # Data Params
        fs: int = 200,
        patch_seconds: float = 1.0,
        overlap_seconds: float = 0.1,
        # Model Params
        embed_dim: int = 512,
        encoder_depth: int = 12,
        encoder_heads: int = 8,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        # Training Params
        mask_ratio: float = 0.55,
        aux_loss_weight: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.aux_loss_weight = aux_loss_weight

        # 1. Input Processing
        self.patch_embed = PatchEmbed(fs, patch_seconds, overlap_seconds, embed_dim)

        # We calculate patch_size and step from the component we just initialized
        self.patch_size = self.patch_embed.patch_size
        self.step = self.patch_embed.step

        # 2. Positional Encoding (Shared between Encoder and Decoder)
        self.pos_enc = PosEnc(n_freqs=4, embed_dim=embed_dim)

        # 3. Encoder
        self.encoder = TransformerEncoderDecoder(embed_dim=embed_dim, depth=encoder_depth, heads=encoder_heads)

        # 4. Decoder (Main Reconstruction)
        self.decoder = MAEDecoder(embed_dim=embed_dim, decoder_depth=decoder_depth, decoder_heads=decoder_heads, patch_size=self.patch_size)

        # 5. Auxiliary Head (Global Token)
        # We concatenate outputs from ALL encoder layers
        self.aux_dim = encoder_depth * embed_dim

        # A learned query vector to look at the encoder outputs
        self.aux_query = nn.Parameter(torch.randn(1, 1, self.aux_dim))
        nn.init.normal_(self.aux_query, std=0.02)

        # Projection: (Depth * Dim) -> Dim
        self.aux_linear = nn.Linear(self.aux_dim, embed_dim, bias=False)

        # Reconstruction Head for Aux Task
        self.aux_predict = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, self.patch_size))

    def prepare_coords(self, xyz: torch.Tensor, num_patches: int):
        B, C, _ = xyz.shape
        device = xyz.device

        # 2. Generate Time Indices (0, 1, 2, ... P-1)
        time_idx = torch.arange(num_patches, device=device, dtype=torch.float32)

        # 3. Expand Spatial Coords
        # (B, C, 3) -> (B, C, 1, 3) -> (B, C, P, 3)
        spat = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)

        # 4. Expand Time Coords
        # (P,) -> (1, 1, P, 1) -> (B, C, P, 1)
        time = time_idx.view(1, 1, num_patches, 1).expand(B, C, -1, -1)

        # 5. Concatenate -> (B, C, P, 4)
        coords = torch.cat([spat, time], dim=-1)

        # 6. Flatten to (B, N_Total, 4)
        return coords.flatten(1, 2)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor):
        B, _, _ = x.shape

        # --- 1. Patchify & Embed ---
        # patches: (B, C, P, PatchSize)
        patches = x.unfold(-1, self.patch_size, self.step)
        num_patches = patches.shape[2]

        # tokens: (B, C, P, Dim)
        tokens = self.patch_embed.linear(patches)

        # Flatten to Sequence: (B, N_Total, Dim)
        tokens_flat = tokens.flatten(1, 2)
        patches_flat = patches.flatten(1, 2)  # Target for loss

        # --- 2. Prepare 4D Coordinates ---
        coords = self.prepare_coords(xyz, num_patches)

        # --- 3. Generate Mask ---
        # Returns mask where counts are GUARANTEED to be equal across batch
        mask = generate_mask(coords, mask_ratio=self.mask_ratio)

        # --- 4. Prepare Encoder Input ---
        # We need to extract only the visible tokens and stack them.
        # Since counts are fixed, we can do this efficiently using boolean masking and reshaping.

        # tokens_flat: (B, N_Total, D)
        # mask: (B, N_Total)
        # Result: (B, N_Vis, D)
        # The .view() works because the number of Trues in mask is identical for every row b.
        n_vis = mask[0].sum().item()

        x_vis = tokens_flat[mask].view(B, n_vis, -1)
        coords_vis = coords[mask].view(B, n_vis, -1)

        # Add PE
        pe_vis = self.pos_enc(coords_vis)
        x_vis = x_vis + pe_vis

        # --- 5. Encoder Forward ---
        x_encoded, intermediates = self.encoder(x_vis)

        # --- 6. Main Decoder Path ---
        predictions_main = self.decoder(x_visible=x_encoded, pos_enc=self.pos_enc, coords=coords, mask=mask)

        # --- 7. Auxiliary Path (Global Token) ---
        # Concatenate all intermediate layers: (B, N_Vis, Depth*Dim)
        aux_input = torch.cat(intermediates, dim=-1)

        # Attention Pooling
        # Score = Input @ Query.T
        # (B, N_Vis, AuxDim) @ (1, 1, AuxDim).T -> (B, N_Vis, 1)
        attn_scores = torch.matmul(aux_input, self.aux_query.transpose(1, 2))
        attn_weights = F.softmax(attn_scores, dim=1)

        # Pool: Sum(Weights * Input) -> (B, 1, AuxDim)
        global_token = torch.sum(attn_weights * aux_input, dim=1, keepdim=True)

        # Project to Embed Dim: (B, 1, Dim)
        global_emb = self.aux_linear(global_token)

        # Predict Masked Patches
        # 1. Get coords of masked tokens
        # Since mask is fixed count, we can reshape cleanly
        n_masked = (~mask[0]).sum().item()
        coords_masked = coords[~mask].view(B, n_masked, -1)

        pe_masked = self.pos_enc(coords_masked)

        # 2. Expand global token
        global_expanded = global_emb.expand(-1, n_masked, -1)

        # 3. Combine & Predict
        aux_pred_in = global_expanded + pe_masked
        predictions_aux = self.aux_predict(aux_pred_in)

        # --- 8. Loss Calculation ---
        # Target: Only the masked patches
        target_masked = patches_flat[~mask].view(B, n_masked, -1)

        # Main Loss (L1 on masked)
        pred_main_masked = predictions_main[~mask].view(B, n_masked, -1)
        loss_main = F.l1_loss(pred_main_masked, target_masked)

        # Aux Loss (L1 on masked)
        loss_aux = F.l1_loss(predictions_aux, target_masked)

        total_loss = loss_main + self.aux_loss_weight * loss_aux

        return total_loss, predictions_main, mask
