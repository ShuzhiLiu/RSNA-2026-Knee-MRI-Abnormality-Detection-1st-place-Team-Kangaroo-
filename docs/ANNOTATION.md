# Annotation: study labels and spatial supervision

The classifiers were trained on **released numeric tables** (supervision dataset
`liushuzhi/rsna-knee-1st-place-training-supervision`). Re-running annotation is optional, uses paid or self-hosted
models, and is not deterministic, so it will not reproduce the released probabilities exactly. This page explains
how each asset was produced and which code in this repository corresponds to it.

| Asset (supervision dataset) | Produced by | Code here |
|---|---|---|
| `labels/astra_v1_weak4349.csv` | GPT-6 Astra, medium reasoning, overview sheets + MRI viewer | `scripts/annotate_astra.py`, `src/rsna_knee/annotation/astra_v1/` |
| `labels/astra_v2_weak4349.csv` | GPT-6 Astra, low reasoning, 24 individual images per fluid series + MRI viewer | `scripts/prepare_final_annotation.py astra-v2-input`, `src/rsna_knee/annotation/astra_v2/` |
| `labels/teammate_astra_weak4349.csv`, `labels/focus_guided_weak4349.csv` | @tascj0 | see his write-up |
| `labels/hulu_weak4349.csv` | Hulu-Med-Flash-Preview-27B, yes/no token probabilities (+ dedicated Synovitis pass) | shared helpers only (`annotation/qwen_next/vlm_data.py`, `vlm_hulumed.py`) |
| `labels/qwen_next_weak4349.csv` | Qwen3.8-Flash-Next-FP8, per-target reasoning then a continuous probability | `src/rsna_knee/annotation/qwen_next/method.py` |
| `labels/mix5_weak4349.csv` | `0.3·v1 + 0.2·v2 + 0.3·teammate + 0.1·Hulu + 0.1·Qwen` | checked by `final_solution.py prepare` |
| `labels/astra111_lesion_eligibility_weak4349.csv` | equal mean of Astra v1, v2, teammate Astra | lesion eligibility (≥ 0.5) only |
| `anatomy/` | GPT-6.1 Sol slice boxes → DINOv3 ConvNeXt-B + DETR detector (2-fold OOF) → state policy | not included (see below) |
| `lesions/` | @tascj0's lesion archive, filtered to eligible training studies | consumed by `src/rsna_knee/box_data.py` |

All label tables cover the same 4,349 report-labelled studies with 12 continuous probabilities in the target order
`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion,
Fracture`. The 58 human-labelled studies always use their original binary labels.

## Astra (agentic GPT reader)

One conversation per study. The model receives:

- the **target definitions** of the competition (high-grade ACL tear, moderate/large effusion, compartment-specific
  OA with ≥50% thickness cartilage loss, …; see `acquire_prompt.txt`);
- the **report projection**: the original report with every clause that mentions synovial terms removed in all
  languages (`astra_v1/project.py`), to avoid anchoring Synovitis on report wording;
- a **series catalog** and, for v1, one overview contact sheet per fluid-sensitive series (`astra_v1/overview.py`);
- an **MRI viewer** exposed as an MCP server (`astra_v1/viewer.py`) with two tools:
  - `view_slices`: 1–4 native slices of one series, a normalized crop box (≥ 0.2 span) and an optional display
    window (original always shown on the left);
  - `compare_location`: a point on a reference slice is mapped through DICOM geometry to candidate neighborhoods
    in 1–3 other series (not verified registration; the model must check anatomy).

  The budget is **10 tool calls and 10 rendered sheets per study**. The prompt requires at least one detailed view
  of each fluid series, one structural view, one successful comparison and two adjusted-window views.

The model returns the JSON in `acquire_schema.json`: for each target a findings summary with image/report evidence,
then `probabilities` — twelve values in [0, 1], estimated independently as P(reference label = 1 | evidence).
Continuous outputs were kept deliberately: on the 58 development studies six-bin outputs scored 0.8832 macro AUC
versus 0.9363 for continuous outputs on the same acquired evidence.

The production call was `codex exec` with a read-only sandbox, an empty working directory, no shell/web/project
docs and the viewer as the only tool (`scripts/annotate_astra.py run` prints the exact argv). Astra v2 used the same
viewer and definitions but a different initial presentation (24 individual 384 px images per fluid series, built by
`astra_v2/build.py`), low reasoning effort, and the Codex app-server protocol instead of `codex exec`.

```bash
python scripts/annotate_astra.py prepare --train-csv $DATA/train.csv --raw-root $DATA \
    --cache-root $WORK/cache --run-dir $WORK/astra --study <uid>          # no model call
python scripts/annotate_astra.py run --case $WORK/astra/cases/<uid>/input.json \
    --attempt $WORK/astra/attempts/<uid>-1 [--execute]                  # --execute = paid call
python scripts/annotate_astra.py collect --attempts $WORK/astra/attempts --output astra_v1.csv

# Astra v2 initial input from the prepared v1 case, and the viewer on its own (stdin/stdout MCP)
python scripts/prepare_final_annotation.py astra-v2-input --study-spec $WORK/astra/cases/<uid>/input.json --output $WORK/astra_v2/<uid>
python scripts/prepare_final_annotation.py astra-viewer --version v1 --case <acquire_case.json> --log <dir>
```

`prepare` decodes any eligible candidate series of the study that is not yet cached (the training `inputs` step only
decodes the five selected series). The runner omits the production run's durable retries, budget accounting and
resume logic. The model name `gpt-6-astra` is the one used during the competition; availability may change.

On the 58 human-labelled studies the retained sources scored (macro AUC, report-assisted): ClinFusion 0.8737,
Hulu 0.9093, Qwen Next 0.8955, Astra v1 0.9364, Astra v2 0.9383. These studies were also used during prompt
development, so these are development numbers. In a matched CoAtNet 224 px student comparison, replacing earlier
multimodal labels with Astra labels moved public AUC from 0.922 to 0.935.

## Hulu and Qwen Next (single-pass readers)

Both read the report together with a fixed packet of 21 slices (7 sagittal, 7 coronal, 7 axial) in one request,
without a viewer. Hulu-Med-Flash-Preview-27B scored each target from constrained yes/no token log-probabilities
(`vlm_data.binary_choice_probability`), thinking disabled; a dedicated Synovitis pass used the same slices at higher
native detail with synovial wording removed. Qwen3.8-Flash-Next-FP8 (self-hosted vLLM, 12 requests per study,
temperature 0, thinking enabled) cites report units by ID, describes image evidence and returns a continuous
probability (`qwen_next/method.py`: `prompt`, `request`, `parse`). The 21-slice packet builder and the Hulu request
driver are not part of this release; these two sources contribute 20% of Mix5.

## Anatomical boxes (6 targets)

1. **Sparse GPT annotation.** GPT-6.1 Sol (low reasoning) read one 448 px slice at a time with view- and
   laterality-specific prompts and returned `[{"bbox_2d": [x1,y1,x2,y2], "label": ...}]` in 0–1000 coordinates for
   the visible anatomical *region* where a condition is assessed (not whether it is present). Example prompt:
   [annotation_examples/anatomy_bbox_prompt_example.txt](annotation_examples/anatomy_bbox_prompt_example.txt).
   Snapshot: 39,415 images from 3,823 studies.
2. **Detector expansion.** DINOv3 ConvNeXt-B + adapter + 6-layer DETR decoder, 512 px / 140 mm canvas, batch 16,
   peak LR 1e-4 cosine, ≤16 epochs, EMA selected by held-out mAP50 (0.952 / 0.944 on the two study-disjoint folds).
   Each fold predicted the studies it did not train on, giving out-of-fold boxes for every slice.
3. **States.** Each slice × class entry is **1 = visible region**, **0 = reliable absence** (only under separately
   checked absence rules) or **−1 = ignore**. Supervision is enabled only for 17 accepted class–role combinations
   (`anatomy/usage_map.json`).

The detector code depends on a third-party detection codebase and is not included; the released
`anatomy/lookup_manifest.json` + NPZ files are exactly what the trainer consumed (4,407 studies, 20,477 series).
Boxes are pixel-edge `[x1,y1,x2,y2]` on the canonical 384×384 / 140 mm canvas, class order
`MCL, MM, LM, MedOA, LatOA, PFOA`.

## Lesion boxes (12 targets)

@tascj0's archive (GPT-6.1 Sol high; 1,800 studies, 3,616 lesions, 111,029 per-slice boxes on the selected
fluid-sensitive series). The released `lesions/lesion_train_index.json` keeps only report-labelled training studies
whose fixed Astra111 eligibility probability is ≥ 0.5 for the finding, with exact series/SOP matching; all 58
human-labelled studies are excluded (1,795 studies remain). Boxes are in **native source-pixel** coordinates and are
mapped to the canvas with `box_geometry.source_to_canvas_affine` using the actual input geometry. Unrequested or
not-visible findings are unknown, not negative; an explicit `present: false` supervises absence within an eligible
case. Only the center slice of a 3-slice window supplies boxes, and all three slices must be valid and undropped.
