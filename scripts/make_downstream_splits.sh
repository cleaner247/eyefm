#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/downstream/disease_binary_base.yaml}

python -m eyemae.make_downstream_splits --config "${CONFIG}"
