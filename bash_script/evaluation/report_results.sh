#!/bin/bash
ROOT="eval_logs"

ALL_EXPS=(
  "base_policy"
)

for EXP_NAME in "${ALL_EXPS[@]}"; do
  python summary_evaluation_libero.py --root ${ROOT} --exp_name ${EXP_NAME}
done
