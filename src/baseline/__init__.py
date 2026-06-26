"""EyeFM baseline module: ML (LR/RF/SVM) + DL (TCN/NST/CNNTransformer/TimesNet).

Quick start:

    cd <repo_root>
    python -m baseline.run_baseline --all \\
        --data-root /path/to/eyemae_fast_dataset_v2 \\
        --out-dir outputs/baseline_$(date +%F)

Reference: docs/eyemae_baseline.md
"""
from . import data_loader, dl_baseline, feature_extraction, ml_baseline, run_baseline

__all__ = [
    "data_loader",
    "dl_baseline",
    "feature_extraction",
    "ml_baseline",
    "run_baseline",
]
