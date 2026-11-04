# RSNA 2026 Knee MRI Abnormality Detection — 1st place (Team Kangaroo)

Code for **sakaku / liushuzhi's three-model group** in Team Kangaroo's first-place solution: study-level label
annotation, data preparation, the anatomy-aware MRI classifier, full-data training and MRI-only inference.

| Submission | Public AUC | Private AUC |
|---|---:|---:|
| Final team ensemble (this group 50% + teammate group 50%) | **0.969** | **0.960** |
| This three-model group alone (post-competition check) | 0.967 | 0.957 |
| MaxViT v2 Base 384, Mix5 labels (post-competition) | 0.962 | 0.952 |
| MaxViT v2 Base 384, focus-guided labels (post-competition) | 0.964 | 0.954 |
| CoAtNet-rMLP-2 384, Mix5 labels (post-competition) | 0.962 | 0.953 |

- Write-up: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752474>
- Teammate @tascj0's write-up (the other half of the ensemble): <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution>
- Final team inference notebook: <https://www.kaggle.com/code/tascj0/knee-final-20261022?scriptVersionId=361034054>
- Released training supervision (labels + boxes): <https://www.kaggle.com/datasets/liushuzhi/rsna-knee-1st-place-training-supervision>

This repository covers our three models only. The teammate's model group is described in his write-up and notebook.

## Solution in one page

1. **Labels (annotation).** 4,349 of the 4,407 training studies had only a radiology report. An agentic GPT reader
   ("Astra") read each study through a bounded MRI viewer (≤10 tool calls / ≤10 image sheets), reconciled the report
   with the images under the competition's target definitions and returned **12 continuous probabilities**. Final
   training labels:
   - **Mix5** = `0.3·Astra v1 + 0.2·Astra v2 + 0.3·teammate Astra + 0.1·Hulu + 0.1·Qwen Next` (same weights for every study/target);
   - **focus-guided Astra** labels from @tascj0 (second Base model).

   The 58 studies with official human labels keep those binary labels. Details: [docs/ANNOTATION.md](docs/ANNOTATION.md).
2. **Spatial supervision (training only).** Anatomical regions for 6 targets (GPT slice annotations expanded to all
   slices by a detector; positive / reliable-negative / ignore states) and lesion boxes for all 12 targets
   (from @tascj0's annotation archive, filtered to eligible training studies). Not needed at inference.
3. **Model.** Five MRI roles (sag/cor/ax fluid-sensitive FS, sag/cor structural), 140 mm physical FOV, 384×384,
   32 slices per role from the central 12–88% depth band → 30 overlapping 3-slice windows per role, a shared 2D
   backbone, per-target window attention, learned role/window embeddings, anatomy-guided residual pooling for six
   targets and a 12-target lesion-evidence head on the same feature map.
4. **Training.** All 4,407 studies, no hold-out, 19 epochs of a 30-epoch OneCycle schedule, batch 8, bf16, EMA 0.995;
   the deployed weights are the uniform average of the EMA states after epochs 15–19.
5. **Inference.** Raw DICOM → same preprocessing → 30 windows per role → probabilities. Our group = mean of the three
   models; final team prediction = 1:1 average with the teammate group.

Architecture and training details: [docs/MODEL_AND_TRAINING.md](docs/MODEL_AND_TRAINING.md).

| Recipe | Backbone (timm, ImageNet weights) | Training labels |
|---|---|---|
| `final_mix5_base384` | `maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k` | Mix5 |
| `final_focus_base384` | `maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k` | focus-guided Astra |
| `final_mix5_coat384` | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | Mix5 |

## Repository layout

```text
scripts/final_solution.py        single entry point: hard-labels, pretrained, inputs, prepare, smoke, train,
                                 export, import-weights, predict, blend
scripts/annotate_astra.py        Astra v1 annotation runner: prepare a case, run codex + MRI viewer, collect CSV
scripts/prepare_final_annotation.py  Astra v2 case builder and the MCP MRI viewer server
configs/                         the three final recipes, the 384 px input pipeline, pinned pretrained manifests
                                 (templates; `prepare` writes the bound copies into a workspace)
assets/final_solution/           image-only series selection for the 4,407 training studies (no labels/reports)
src/rsna_knee/                   preprocessing, input pipeline, model, losses, trainer, inference
src/rsna_knee/annotation/        Astra v1/v2 viewer, prompts and output schema; Qwen Next / Hulu request code
tests/                           CPU unit tests
docs/                            annotation, model and training notes
```

## Requirements

- Linux, Python 3.12, an NVIDIA GPU with CUDA, and `g++` (a small C++ polygon kernel for lesion masks is compiled on
  first use into `$RSNA_KERNEL_CACHE` or the system temp dir).
- `pip install -r requirements.txt` (install the CUDA build of torch/torchvision that matches your driver first).
  The original environment: torch 2.13.0+cu130, torchvision 0.28.0, timm 1.0.28, pydicom 3.0.2.
- Training at batch 8 with activation checkpointing peaked at about **44 GiB** (MaxViT v2 Base) and **11 GiB**
  (CoAtNet, 32-image backbone chunks). On one RTX PRO 6000 Blackwell (96 GB) a full fit took **10.8 h** (each Base
  model) and **12.9 h** (CoAtNet).
- Disk: about **436 GiB** for the decoded series cache of the selected series plus **~190 GiB** for the 384 px
  stage cache.

## Data

1. **Competition data** (accept the competition rules): a directory containing `train.csv`, `train_series.csv`,
   `train_series/<study>/<series>/*.dcm`, `test.csv`, `test_series.csv` and `test_series/...`.
2. **Training supervision** (labels, anatomical lookup, lesion index):
   ```bash
   kaggle datasets download -d liushuzhi/rsna-knee-1st-place-training-supervision -p $SUP --unzip
   (cd $SUP && sha256sum -c --quiet SHA256SUMS)
   ```
   The directory must contain `labels/`, `anatomy/`, `lesions/` and `reference/`. Official human labels are not
   redistributed; they are read from your `train.csv`.
3. **Trained weights** (optional, to skip training): one Kaggle dataset per model —
   `liushuzhi/knee-fullmix5-b384swa-20261021`, `liushuzhi/knee-fullfg-b384swa-20261021`,
   `liushuzhi/knee-fullmix5-coat384swa-20261021` (see "Inference with the released weights").

## Reproduce training

```bash
export DATA=/path/to/rsna-knee-abnormality-detection   # competition data
export SUP=/path/to/rsna-knee-1st-place-training-supervision
export WORK=/path/to/work                               # large disk
PY=python

# 1. The 58 official human-labelled studies (binary labels, kept as-is in training)
$PY scripts/final_solution.py hard-labels --train-csv $DATA/train.csv --output $WORK/h58.csv

# 2. Pinned ImageNet backbones from Hugging Face (revision + SHA256 checked)
$PY scripts/final_solution.py pretrained --output-root $WORK/pretrained

# 3. Image-only series selection for the 4,407 training studies
gunzip -c assets/final_solution/training_selection.jsonl.gz > $WORK/selection.jsonl

# 4. Decode the selected DICOM series and build the canonical 384 px / 140 mm stage cache (several hours)
$PY scripts/final_solution.py inputs --raw-root $DATA --selection $WORK/selection.jsonl \
    --cache-root $WORK/cache --stage-cache-root $WORK/stage_cache

# 5. Bind supervision, human labels, caches and backbones into a new workspace (checks counts, IDs,
#    the Mix5 formula, lesion/anatomy hashes and pretrained hashes)
$PY scripts/final_solution.py prepare --supervision $SUP --hard-labels $WORK/h58.csv \
    --selection $WORK/selection.jsonl --cache-root $WORK/cache --stage-cache-root $WORK/stage_cache \
    --base-pretrained $WORK/pretrained/maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k \
    --coat-pretrained $WORK/pretrained/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
    --output $WORK/workspace

# 6. Optional: a few optimizer steps to check the whole pipeline (~1 min; the result is NOT a usable model)
$PY scripts/final_solution.py smoke --workspace $WORK/workspace --recipe final_mix5_base384 \
    --output $WORK/smoke_base --steps 6

# 7. Full fits (run one at a time; the trainer refuses to start on an occupied GPU)
for R in final_mix5_base384 final_focus_base384 final_mix5_coat384; do
  $PY scripts/final_solution.py train --workspace $WORK/workspace --recipe $R --output $WORK/runs/$R
  $PY scripts/final_solution.py export --workspace $WORK/workspace --recipe $R \
      --run $WORK/runs/$R --output $WORK/packages/$R
done
```

`train` snapshots the code and prepared configs into the run directory, trains 19 epochs (550 steps each), saves the
EMA state after epochs 15–19 and writes `swa/model_swa.pt` (their uniform float64 average, no BN recalibration).
The backend is deterministic (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic cuDNN and algorithms), but a fresh fit
on different hardware/software is not guaranteed to be bit-identical to ours.

## Predict

```bash
# our three-model group (mean of the three members); --data needs test.csv, test_series.csv and test_series/
$PY scripts/final_solution.py predict \
    --packages $WORK/packages/final_mix5_base384 $WORK/packages/final_focus_base384 $WORK/packages/final_mix5_coat384 \
    --data $DATA --output $WORK/pred/group.csv --precision fp16

# or per member, then the final 1:1 team blend with the teammate group's prediction CSV
$PY scripts/final_solution.py blend --members m1.csv m2.csv m3.csv --teammate teammate.csv --output team.csv
```

Inference needs only the MRI DICOMs: no reports, annotation files or detector. `predict` also writes
`<output>.members.json` with each member's probabilities.

### Inference with the released weights

```bash
kaggle datasets download -d liushuzhi/knee-fullmix5-b384swa-20261021 -p $WORK/dl/mix5_base384 --unzip
$PY scripts/final_solution.py import-weights --source $WORK/dl/mix5_base384 --output $WORK/packages/final_mix5_base384
# same for knee-fullfg-b384swa-20261021 and knee-fullmix5-coat384swa-20261021, then `predict` as above
```

`import-weights` checks `state.pt` / `pretrained.safetensors` against the dataset's `PROVENANCE.json` and writes a
predictor package (`configs/run.json`, `configs/pretrained.json`, `pretrained/`, `state.pt`, `PACKAGE.json`).

## Annotation

Re-annotation is optional: the classifiers were trained on the released numeric tables. The code is included to
document and re-run the method. See [docs/ANNOTATION.md](docs/ANNOTATION.md).

```bash
# build one Astra v1 case (series selection, overview sheets, report projection, viewer manifest); no model call
$PY scripts/annotate_astra.py prepare --train-csv $DATA/train.csv --raw-root $DATA \
    --cache-root $WORK/cache --run-dir $WORK/astra --study <StudyInstanceUID>
# print the codex command; add --execute to run it (paid, needs Codex CLI with access to the model)
$PY scripts/annotate_astra.py run --case $WORK/astra/cases/<StudyInstanceUID>/input.json --attempt $WORK/astra/attempts/<id>
$PY scripts/annotate_astra.py collect --attempts $WORK/astra/attempts --output astra_v1.csv
```

## What was verified for this release (2026-11-03)

Run from this directory on one RTX PRO 6000 Blackwell, with the public supervision dataset and the competition data:

- CPU unit tests pass (`python -m unittest discover tests`).
- `pretrained` downloaded both backbones at the pinned revisions; sizes and SHA256 match.
- The supervision package passes `sha256sum -c` (20,500 files). `prepare` produced training targets identical to the
  frozen records of the original runs for all 4,407 studies (same order, both label recipes); the prepared recipes
  differ from the original runs' resolved configs only in data paths and relocated-manifest hashes.
- `inputs` rebuilt the caches of two studies from raw DICOM: every pixel volume is byte-identical to the caches the
  final models were trained on (metadata files additionally record source-file counts).
- `smoke` (6 optimizer steps) ran for all three recipes: finite and decreasing loss, anatomical and lesion auxiliary
  losses active; peak memory 43.8 GiB (Base) / 11.0 GiB (CoAtNet); ~3.6–4.0 s per step. Repeated runs gave the same
  first-step loss.
- `train` (code snapshot, frozen re-execution, full-data loader with 550 steps/epoch) was started from a fresh copy of
  this directory and stopped after 21 optimizer steps; `predict` and `blend` ran on the smoke packages.
- `import-weights` + `predict` with the three original winning models reproduced the probabilities of the deployed
  competition predictor exactly (max abs difference 1e-16, fp16 and fp32, 8 studies decoded from raw DICOM).
- The Astra v1 and v2 case builders reproduced the original production inputs for a test study exactly (prompt text,
  report projection, image hashes); the MCP viewer answered `view_slices` / `compare_location` requests.

Not re-run: a complete 19-epoch fit, the paid annotation model calls, and the anatomical detector.

## Attribution and license

Astra v1/v2 labels, Hulu/Qwen labels, anatomical boxes, the classifier and this code: sakaku / liushuzhi.
Teammate-Astra and focus-guided labels and the source lesion annotations: @tascj0 (his releases keep their own terms).

This code is released under the **MIT License** (see [LICENSE](LICENSE)). The released supervision dataset remains
under **CC BY-NC 4.0**. Third-party components keep their licenses (timm and the pretrained weights: Apache-2.0). Competition data
are subject to the competition rules and are not redistributed here. Model outputs and annotations are research
artifacts, not clinical diagnoses.
