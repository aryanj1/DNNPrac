#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -ne 0 ]]; then
  echo 'run.sh uses the default configuration. For custom settings use python -m cyclegan train/infer (see README.md).'
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo 'First follow the environment setup in README.md.'
  exit 1
fi
if [[ ! -d data/apple2orange/trainA ]]; then
  .venv/bin/python scripts/download_data.py
fi
if [[ -f runs/apple2orange/latest.pt ]]; then
  .venv/bin/python -m cyclegan train --resume runs/apple2orange/latest.pt --epochs 10 --decay-start 5
else
  .venv/bin/python -m cyclegan train
fi
.venv/bin/python -m cyclegan infer --checkpoint runs/apple2orange/latest.pt --direction ab --input data/apple2orange/testA --output outputs/apple2orange
.venv/bin/python -m cyclegan infer --checkpoint runs/apple2orange/latest.pt --direction ba --input data/apple2orange/testB --output outputs/orange2apple
