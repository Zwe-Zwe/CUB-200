#!/usr/bin/env bash
# Re-run the complete Track 2 ladder (L0 to L7, plus the L8 checks) for both models from scratch, with the same
# configs and seed, into outputs_rerun/ so the original runs in outputs/ are not touched. Then evaluate everything.
# Purpose: check that the reported results reproduce. About 3.5 hours on an RTX 4060 laptop GPU.
#
#   bash scripts/rerun_ladder.sh            (log: outputs_rerun/rerun.log)
cd "$(dirname "$0")/.." || exit 1
export BIRDCLS_OUTPUT_DIR=outputs_rerun
PY=${PY:-python}
mkdir -p outputs_rerun
log() { echo "[$(date '+%F %T')] $*"; }

R=resnet50.tv2_in1k
X=convnext_tiny
log "start"
$PY -m birdcls.train --model $R --config tl_L0_probe tl_L1_finetune tl_L2_discriminative tl_L3_regularise \
    tl_L4_strongaug tl_L5_longer_strongaug tl_L6_hires_strongaug || log "ResNet ladder: a run failed"
$PY -m birdcls.train --model $R --config tl_L6_hires_strongaug --full-train || log "ResNet final failed"
$PY -m birdcls.train --model $X --config tl_L0_probe tl_L1_finetune tl_L2_discriminative tl_L3_regularise \
    tl_L4_strongaug tl_L5_longer tl_L6_hires || log "ConvNeXt ladder: a run failed"
$PY -m birdcls.train --model $X --config tl_L6_hires --full-train || log "ConvNeXt final failed"
$PY -m birdcls.train --model $R --config tl_L8_sgd_1e-3 tl_L8_sgd_3e-3 tl_L8_sgd_1e-2 || log "ResNet L8 failed"
$PY -m birdcls.train --model $X --config tl_L8_llrd_080 tl_L8_llrd_065 || log "ConvNeXt L8 failed"
log "training done, evaluating"
$PY -m birdcls.evaluate || log "evaluate failed"
log "done"
