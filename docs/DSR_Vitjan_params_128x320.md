# Vitjan et al. DSR → 128×320 log-mel spectrograms

This document maps the **Vitjan et al. DSR (Dual Subspace Reconstruction)** VQ-VAE (MVTec images) to a **two-level architecture for 128×320 log-mel spectrograms**, and lists the parameters you need for encoder/decoder layers.

---

## Fig. 1 (AudDSR diagram) ↔ Vitjan shallower blocks

The AudDSR diagram (Fig. 1) has the same **data flow** as Vitjan’s DSR; the difference is that Vitjan uses **shallower** encoder/decoder blocks (one ResidualStack per stage, no extra down/up convs between stacks). Here is how each block in the diagram maps to a Vitjan-style implementation.

| Fig. 1 block | Role | Vitjan-style (shallower) implementation |
|--------------|------|----------------------------------------|
| **X** | Input | (B, 1, 128, 320) log-mel. |
| **Encoder 1** | X → f1 | **EncoderBot:** 2× Conv2d(4, stride=2, pad=1) → 4× down; then Conv2d(3, stride=1); then **one** ResidualStack. No second down-conv or second ResStack. Output f1 at (32, 80) with `num_hiddens` channels. Optionally 1×1 to `latent_dim` for AudDSR. |
| **Encoder 2** | f1 → f2 | **EncoderTop:** 1× Conv2d(4, stride=2, pad=1) → 2× down; then Conv2d(3, stride=1); then **one** ResidualStack. Output f2 at (16, 40) with `num_hiddens` channels. Optionally 1×1 to `latent_dim`. |
| **Pre-VQ (top)** | f2 → latent for VQ1 | 1×1 conv: `num_hiddens → embedding_dim` (if encoders output num_hiddens). If encoders already output `latent_dim`, this can be identity. |
| **VQ1** | Coarse codebook | Unchanged: VectorQuantizerEMA → Q1 at (16, 40). |
| **General Decoder Module 1** | Q1 → fU | **Shallow (Vitjan):** 1×1 conv → **one** ResidualStack → 2× ConvTranspose2d. Mirror of encoder: no second ResStack. |
| **Concat [f1, fU]** | Fine feature for VQ2 | Unchanged. If f1 is num_hiddens, pre_vq2: (num_hiddens + latent_dim) → latent_dim. |
| **Pre-VQ (bottom)** | → zb | 1×1: (num_hiddens + embedding_dim) → embedding_dim, or (2*latent_dim) → latent_dim if f1 is already latent_dim. |
| **VQ2** | Fine codebook | Unchanged → Q2 at (32, 80). |
| **Q1 upsampled** | To Q2 resolution | Bilinear interpolate Q1 to (32, 80); optional 1×1. Unchanged. |
| **Concat [Q1_up, Q2]** | Decoder 2 input | Unchanged: 2×embedding_dim at (32, 80). |
| **General Decoder Module 2** | → X_out | **DecoderBot (Vitjan):** 1×1 conv → **one** ResidualStack → 2× ConvTranspose2d. Same shallow pattern as Decoder 1. Output channels = **1** for log-mel. |

So the transfer is: **keep the diagram’s flow and loss**; replace each “Encoder” and “Decoder module” with the **shallower** Vitjan block (one ResidualStack per block, fixed stride-2 convs for down/up).

---

## 1. Spatial flow comparison

| Stage | Vitjan (e.g. 64×64 or 128×128 image) | Your 128×320 log-mel |
|-------|--------------------------------------|----------------------|
| **Input** | (B, 3, H, W) | (B, 1, 128, 320) |
| **EncoderBot (fine)** | 2× conv 4×4 stride 2 → (H/4, W/4) | 2× stride 2 or 1× stride 4 → **(32, 80)** |
| **EncoderTop (coarse)** | 1× conv 4×4 stride 2 → (H/8, W/8) | 1× stride 2 → **(16, 40)** |
| **VQ top** | on (H/8, W/8) | on (16, 40) |
| **Upsample top** | 1× ConvTranspose 4×4 stride 2 → (H/4, W/4) | 1× stride 2 → (32, 80) |
| **VQ bottom** | on (H/4, W/4), after concat [enc_b, up_Q_top] | on (32, 80) |
| **Decoder** | input (H/4, W/4), 2× ConvTranspose stride 2 → (H, W) | (32, 80) → 2× stride 2 → **(128, 320)** |
| **Output** | (B, 3, H, W) | (B, 1, 128, 320) |

So for 128×320 you keep the **same two-level hierarchy**: fine latent at **32×80**, coarse at **16×40**.

---

## 2. Parameters to set for encoder/decoder layers

### 2.1 Input / output

| Parameter | Vitjan (MVTec) | Your 128×320 log-mel |
|-----------|----------------|----------------------|
| `in_channels` | 3 (RGB) | **1** (mono spectrogram) |
| Decoder output channels | 3 (RGB) | **1** (reconstructed log-mel) |

### 2.2 EncoderBot (fine encoder: X → enc_b)

Vitjan uses:

- **Conv1**: `in_channels → num_hiddens//2`, kernel **4**, stride **2**, padding **1**
- **Conv2**: `num_hiddens//2 → num_hiddens`, kernel **4**, stride **2**, padding **1**
- **Conv3**: `num_hiddens → num_hiddens`, kernel **3**, stride **1**, padding **1**
- **ResidualStack**: `num_hiddens`, `num_residual_layers`, `num_residual_hiddens`

For 128×320 you need **4× down** in both dimensions (128→32, 320→80). So either:

- **Option A (match Vitjan exactly):** two convs with kernel 4, stride 2, padding 1 (same as above).
- **Option B (your current style):** one conv with kernel (4,4) or (8,8) and stride (4,4).

Parameters to fix:

- `num_hiddens` — e.g. **128** (Vitjan-style) or **64** (lighter). Your current `enc1_hidden` is this role.
- `num_residual_layers` — Vitjan often uses **2** or **3**; your `num_res_blocks`.
- `num_residual_hiddens` — middle channels in each residual block; e.g. **32**. Your current blocks use the same channel count; if you add Vitjan-style ResidualStack, set this.

### 2.3 EncoderTop (coarse encoder: enc_b → enc_t)

Vitjan uses:

- **Conv1**: `num_hiddens → num_hiddens`, kernel **4**, stride **2**, padding **1**
- **Conv2**: `num_hiddens → num_hiddens`, kernel **3**, stride **1**, padding **1**
- **ResidualStack**: same as above

For 128×320, **2× down** from (32, 80) → (16, 40). So one stride-2 conv is enough.

Parameters:

- Same `num_hiddens`, `num_residual_layers`, `num_residual_hiddens` as EncoderBot (or a bit smaller if you want).

### 2.4 Pre-VQ convolutions

- **Top (coarse):** `num_hiddens → embedding_dim`, 1×1 conv.  
  Your `pre_vq_conv_top` / latent projection for f2.
- **Bottom (fine):** input is `[enc_b, up_quantized_t]` so channels = `num_hiddens + embedding_dim` → **embedding_dim**, 1×1 conv.  
  Your `pre_vq2`: `latent_dim * 2 → latent_dim`.

So:

- `embedding_dim` = your **latent_dim** (e.g. **64** or **128**).
- Top: `enc2_hidden` or `num_hiddens` → `latent_dim`.
- Bottom: `num_hiddens + latent_dim` → `latent_dim`.

### 2.5 DecoderBot (single decoder: [Q_top_up, Q_bot] → X_recon)

Vitjan uses:

- **Conv1**: `embedding_dim*2 → num_hiddens`, kernel **3**, stride **1**, padding **1**
- **ResidualStack**: same `num_hiddens`, `num_residual_layers`, `num_residual_hiddens`
- **ConvTranspose1**: `num_hiddens → num_hiddens//2`, kernel **4**, stride **2**, padding **1**
- **ConvTranspose2**: `num_hiddens//2 → out_channels`, kernel **4**, stride **2**, padding **1**

For 128×320 the decoder input is at (32, 80); you need **4× up** to (128, 320). So two stride-2 transposed convs, same as Vitjan.

Parameters:

- `in_channels` for decoder = **embedding_dim * 2** (concat of upsampled top code and bottom code).
- `out_channels` = **1** (not 3).
- Same `num_hiddens`, `num_residual_layers`, `num_residual_hiddens` as encoders.

### 2.6 VQ and loss

- **num_embeddings** (codebook size): e.g. **512** or **256**.
- **embedding_dim**: same as latent dimension, e.g. **64** or **128**.
- **commitment_cost** (Vitjan) = your **lambda_k**: e.g. **0.25**.
- **decay**: EMA decay for codebook, e.g. **0.99**.

---

## 3. Suggested parameter set for 128×320 (Vitjan-like)

Use these as a single place to tune:

```python
# Spatial (fixed by 128×320 and 4× / 2× design)
enc_b_shape = (32, 80)   # fine
enc_t_shape = (16, 40)   # coarse

# Channels and capacities
in_channels = 1
out_channels = 1
num_hiddens = 128          # encoder/decoder hidden (Vitjan uses 128)
embedding_dim = 64          # VQ latent dim
num_residual_layers = 2     # or 3
num_residual_hiddens = 32   # inside each residual block

# VQ
num_embeddings = 512       # or 256
commitment_cost = 0.25     # = lambda_k
decay = 0.99
```

Encoder/decoder layer checklist:

- **EncoderBot:** two convs 4×4 stride 2 pad 1 (→ 32×80), then 3×3 stride 1, then ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens).
- **EncoderTop:** one conv 4×4 stride 2 pad 1 (→ 16×40), then 3×3 stride 1, then ResidualStack.
- **Pre-VQ top:** 1×1 conv num_hiddens → embedding_dim.
- **Pre-VQ bottom:** 1×1 conv (num_hiddens + embedding_dim) → embedding_dim.
- **Upsample top:** ConvTranspose2d(embedding_dim, embedding_dim, 4, stride=2, padding=1).
- **DecoderBot:** Conv 3×3 (embedding_dim*2 → num_hiddens), ResidualStack, ConvTranspose 4 stride 2 (num_hiddens → num_hiddens//2), ConvTranspose 4 stride 2 (num_hiddens//2 → out_channels=1).

---

## 4. Mapping to your current AudDSR

Your `AudDSR` already implements the same two-level idea with `Encoder2d`, two VQs, and two decoder modules. To align with Vitjan’s layer roles and capacities:

- **enc1_hidden** = `num_hiddens` (e.g. 128); **enc2_hidden** = same or slightly smaller.
- **latent_dim** = `embedding_dim` (e.g. 64).
- **enc1_downsample** = **(4, 4)** so 128×320 → 32×80 (replaces Vitjan’s two 2× convs with one 4×).
- **enc2_downsample** = **(2, 2)** so 32×80 → 16×40.
- **dec1_hidden / dec2_hidden** = `num_hiddens` (e.g. 128).
- **num_res_blocks** = `num_residual_layers` (e.g. 2 or 3).
- **lambda_k** = `commitment_cost` (0.25).

Your residual blocks use GroupNorm + dilated convs; Vitjan uses ReLU-first, no dilation, and `num_residual_hiddens`. If you want to match Vitjan’s residual design exactly, use the `Residual` and `ResidualStack` in `dsr_vitjan_style.py` (see below) and set `num_residual_hiddens` (e.g. 32) there.

---

## 5. Summary table (parameters to take into account)

| Parameter | Role | Suggested (128×320) |
|-----------|------|---------------------|
| **in_channels** | Input channels | 1 |
| **out_channels** | Reconstruction channels | 1 |
| **num_hiddens** | Encoder/decoder hidden width | 128 |
| **embedding_dim** | VQ latent dim (and codebook dim) | 64 |
| **num_residual_layers** | Depth of residual stack | 2 or 3 |
| **num_residual_hiddens** | Mid channels in residual block | 32 |
| **num_embeddings** | Codebook size | 512 or 256 |
| **commitment_cost / lambda_k** | Commitment loss weight | 0.25 |
| **decay** | EMA decay for VQ | 0.99 |
| **Kernel / stride / pad (EncBot)** | 4×4, stride 2, pad 1 (×2) then 3×3 stride 1 | — |
| **Kernel / stride / pad (EncTop)** | 4×4, stride 2, pad 1 then 3×3 stride 1 | — |
| **Decoder** | 3×3 stride 1, then 4×4 stride 2 (×2), last out_channels=1 | — |

All of these are the parameters you need to take into account to follow a Vitjan-like two-level architecture on 128×320 log-mel spectrograms.
