# EXP-23: Full re-run of the Track 2 ladder (reproducibility check)

**Status:** done 2026-10-21 (02:38 to 06:27, run unattended at the user's request) · **Script:**
`scripts/rerun_ladder.sh` (outputs in `outputs_rerun/`, originals in `outputs/` untouched) · **Comparison:**
`scripts/compare_rerun.py` → `outputs_rerun/comparison.csv`

## Why
Before submitting, the user asked to retrain every step from L0 to the end for both models, to make sure the
reported results are trustworthy and not a one-off.

## What was run
The same 21 configurations as EXP-13 to EXP-21, same code, same seed (42), same data split, from the pretrained
weights again: ResNet-50 L0 to L6 + final (`--full-train`) + 3 L8 SGD runs; ConvNeXt-Tiny L0 to L6 + final + 2 L8
layer-decay runs. Then `python -m birdcls.evaluate` on all of them. All outputs were redirected with the new
`BIRDCLS_OUTPUT_DIR=outputs_rerun` setting (`birdcls/config.py`).

## Result: bit-for-bit reproduction

| Check | Result |
|---|---|
| Runs finished | 21 / 21, no errors |
| Test Top-1, average per class, validation accuracy | identical in all 21 runs (max difference 0.00 pp) |
| Best epoch (checkpoint kept) | identical in all 21 runs |
| Per-image test predictions | identical for all 1,204 test images in all 21 runs |

Final models again: ConvNeXt-Tiny 90.20% / 89.94%, ResNet-50 86.46% / 86.12%.

## Interpretation
The whole pipeline (data split, augmentation order, initialisation of the new head, training, evaluation) is fully
deterministic given the seed, so every number in the report can be regenerated exactly from the code and configs.
This shows the results are not a lucky run of a non-repeatable process. It does **not** measure seed-to-seed variance
(a different seed would change the split, the augmentation and the head initialisation); that remains the
"single seed" limitation in the report, covered by the McNemar tests and the bootstrap intervals (Appendix A.2, A.3).
