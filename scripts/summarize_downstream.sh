#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-outputs/downstream_disease_binary_seed42}

python -m eyemae.summarize_downstream --output_root "${ROOT}"
