#!/usr/bin/env bash
set -euo pipefail

# 32 physical cores / 64 GB RAM CPU-only run.
# GPU and embeddings are intentionally NOT used in this stage.
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

python entity_resolution_pipeline_cpu_max.py \
  --train-dir dataset/train \
  --test-dir dataset/test \
  --output-dir output \
  --work-dir work \
  --validator utils/validate_submission.py \
  --chunk-size 250000 \
  --cache-gb 12 \
  --mmap-gb 32 \
  --max-pos 7500000 \
  --max-neg 8000000 \
  --singleton-neg 2000000 \
  --max-train-rows 14000000 \
  --n-estimators 1200 \
  --keep-work
