#!/bin/bash

GPU=0

OUTDIR="eval_logs"

POLICY_CKPT=""

for SEED in 1; do
CUDA_VISIBLE_DEVICES=${GPU} python eval_libero.py \
  --checkpoint ${POLICY_CKPT} \
  --output_dir "${OUTDIR}" \
  --seed ${SEED}
done
