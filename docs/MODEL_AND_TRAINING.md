# Model and training

The three final recipes (`configs/final_*.json`) share everything except the backbone and the training-label table.
`src/rsna_knee/final_config.py` refuses recipes that change the architecture, input view or auxiliary losses.

## Input pipeline

| Step | Implementation |
|---|---|
| DICOM decode, rescale, MONOCHROME1 handling, physical slice ordering, 0.5/99.5 percentile clip, uint16 cache + geometry metadata | `preprocessing.py` |
| Series selection for the five roles (`SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_STRUCT, COR_STRUCT`) | `series_selection.py` (training: released `training_selection.jsonl.gz`; inference: `inference/raw_decode.py`) |
| Laterality, canonical in-plane orientation, 140 mm physical center crop/pad, 384×384 bilinear-antialias resize, deterministic stage cache | `input_pipeline.py` |
| Central depth band [0.12, 0.88] → 32 slices per role → 30 overlapping 3-slice windows; masks for missing roles, repeated/invalid slices and padded pixels | `model_input.py`, `inference/raw32_input.py` |

Each 3-slice window is one 3-channel image for the 2D backbone; different sequences are never stacked as channels.
Training samples 8 of the 30 windows per role; inference uses all 30. The input config keeps the historical cache
field `slices_per_series: 16`; the model view is defined by `derived_model_view` (depth 32, 30 windows).

Training-only augmentation (validation/inference unchanged): GPU per-window affine (rotation ±16°, zoom-in 0.16,
shift 0.1, intensity gain 0.2, no flips); CPU per-role affine on the 32-slice view (rotation ±7°, translation 0.04,
scale 0.95–1.05), contrast/gamma 0.9–1.1, Gaussian noise ≤ 0.02, slice dropout 0.04, block dropout 0.15 (≤ 2 slices),
role dropout 0.04 (never the last available role); depth-center jitter and depth-band scale ∈ [0.8, 1.25]. Boxes
and masks are transformed with the images.

## Architecture (`dense_window_model.py`, `location_pooling.py`, `box_model.py`)

- **Backbone:** `maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k` or `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (timm,
  pinned revision and SHA256), all layers trainable, activation checkpointing with BatchNorm statistics preserved on
  recompute. CoAtNet runs the backbone in chunks of 32 images.
- **Window features:** stage-4 feature map → global average `g`. For the six location-dependent targets (MCL, medial
  and lateral meniscus, medial/lateral/patellofemoral OA) a target-specific spatial attention readout `a` is mixed in
  residually: `h = g + γ·(a − g)` with learnable `γ` initialised at 0.1. The other six targets use `g`.
- **Fusion:** features projected to 256-d plus learned role and window-position embeddings and a projection of four
  window geometry values (mean normalized depth, depth span, valid-slice fraction, non-repeated fraction). Each target
  attends over the windows of each role; the five role vectors are concatenated (1,280-d) and passed to a per-target
  head (256 hidden, dropout 0.2). Missing roles are masked. Head biases start at the logit of the 4,349 weak-label
  prevalences.
- **Lesion-evidence head:** per-target score maps on the same feature map, normalized log-sum-exp over space and
  valid windows, mean over available roles → `e`. Final logits `z = z_main + 0.25·(e − logit(p_train))`, where
  `p_train` is the weak-label training prevalence. No extra backbone pass and no annotation input at inference.

## Losses

- Main: masked BCE with soft targets (S4349 probabilities, H58 binary) + 0.05 × pairwise soft ranking loss
  (softplus on logit differences, pairs with target gap ≥ 0.25, weighted by |Δq|, averaged over active targets).
- Anatomical auxiliaries (`location_pooling.py`), each 0.1 × w: regional MIL inside visible-structure boxes,
  attention-mass hinge (mass inside the box ≥ 0.7), and negative-window suppression (mass ≤ 0.1 where the structure
  is reliably absent). Ignored entries contribute nothing.
- Lesion MIL (`box_model.py`): 0.1 × w, BCE on log-sum-exp pooled lesion logits inside the union of center-slice boxes.
- Shared schedule with zero-based epoch `e` and in-epoch progress `s`:
  `w = s` (e = 0), `1` (1 ≤ e ≤ 9), `1 − 0.7·(e − 10)/19` (e ≥ 10).

## Optimization (identical for all three recipes)

| Setting | Value |
|---|---|
| Data | all 4,407 studies (S4349 soft + H58 binary), batch 8, drop_last → 550 steps/epoch |
| Optimizer | AdamW, weight decay 0.02, betas (0.9, 0.999), fused, grad-norm clip 1.0 |
| LR | backbone peak 8.944e-5 with stage-wise decay 0.9; new layers 1e-3 |
| Schedule | OneCycle over a 30-epoch horizon (16,500 steps), pct_start 0.15, cosine, div 25, final div 1e4, momentum 0.85–0.95 |
| Stop | fixed after 19 epochs (10,450 steps); no validation, no early stopping |
| EMA | decay 0.995 (`torch.optim.swa_utils.AveragedModel`, buffers included) |
| Final weights | uniform float64 average of EMA after epochs 15–19, cast back; no BN recalibration |
| Precision / backend | bf16 autocast, TF32, deterministic cuDNN/algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed 20260823 |

Inference packages run in fp16 by default (`--precision bf16|fp32` available).

## Development protocol (not used for the final fits)

Late-stage comparisons used one fixed fold: 435 report-labelled studies held out, 3,914 + 58 for training, scored with
probability-weighted SoftAUC against the equal Astra v1/v2/teammate blend, best EMA by SoftAUC (min 10 / max 30 epochs,
patience 5). The final models were refit on all data with the fixed 19-epoch rule above, so they have no hold-out
score; their public/private scores are listed in the README.
