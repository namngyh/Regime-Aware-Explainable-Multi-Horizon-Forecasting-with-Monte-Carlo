#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-raemf}"
export PYTHONPATH="${PYTHONPATH:-src}"
conda run -n eda python -m raemf_mc.cli run --data data.csv --config configs/research.yaml
