from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-eyemae")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_visualizations(
    batch: dict[str, Any],
    pred,
    mae_mask,
    out_dir: str | Path,
    cfg: dict[str, Any],
    *,
    max_trials: int = 16,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    content = batch["content"].detach().cpu().numpy()
    quality = batch["quality"].detach().cpu().numpy()
    stim = batch["stim"].detach().cpu().numpy()
    eye_token_valid = batch["eye_token_valid"].detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    mask_np = mae_mask.detach().cpu().numpy()
    nplot = min(max_trials, content.shape[0])
    for b in range(nplot):
        n = int((~batch["pad_mask"][b]).sum().item())
        t = np.arange(n * content.shape[3])
        for eye, eye_name in enumerate(("left", "right")):
            true_xy = content[b, :n, eye, :, 0:2].reshape(-1, 2)
            pred_xy = pred_np[b, :n, eye, :, 0:2].reshape(-1, 2)
            blink_true = content[b, :n, eye, :, 3].reshape(-1)
            blink_prob = 1.0 / (1.0 + np.exp(-pred_np[b, :n, eye, :, 3].reshape(-1)))
            mask = np.repeat(mask_np[b, :n, eye], content.shape[3])
            missing = quality[b, :n, eye, :, 0].reshape(-1) > 0.5
            invalid = np.repeat(~eye_token_valid[b, :n, eye], content.shape[3])
            fix_on = stim[b, :n, :, 0].reshape(-1)
            stim_on = stim[b, :n, :, 1].reshape(-1)
            stim_x = stim[b, :n, :, 2].reshape(-1)
            stim_y = stim[b, :n, :, 3].reshape(-1)
            fig, axes = plt.subplots(5, 1, figsize=(11, 8.5), sharex=True)
            axes[0].plot(t, true_xy[:, 0], label="true x", linewidth=0.8)
            axes[0].plot(t, pred_xy[:, 0], label="pred x", linewidth=0.8)
            axes[0].plot(t, stim_x, label="stim x", linewidth=0.7, linestyle="--")
            axes[1].plot(t, true_xy[:, 1], label="true y", linewidth=0.8)
            axes[1].plot(t, pred_xy[:, 1], label="pred y", linewidth=0.8)
            axes[1].plot(t, stim_y, label="stim y", linewidth=0.7, linestyle="--")
            axes[2].plot(t, blink_true, label="blink true", linewidth=0.8)
            axes[2].plot(t, blink_prob, label="blink pred", linewidth=0.8)
            axes[3].plot(t, mask.astype(float), label="mae mask", linewidth=0.8)
            axes[3].plot(t, missing.astype(float), label="missing", linewidth=0.8)
            axes[3].plot(t, invalid.astype(float), label="invalid eye token", linewidth=0.8)
            axes[4].plot(t, fix_on, label="fix_on", linewidth=0.8)
            axes[4].plot(t, stim_on, label="stim_on", linewidth=0.8)
            for ax in axes:
                ax.legend(loc="upper right", fontsize=7)
            fig.suptitle(f"{batch['subject_id'][b]} {batch['trial_id'][b]} {eye_name} task={int(batch['task_id'][b])}")
            fig.tight_layout()
            fig.savefig(out / f"{b:02d}_{eye_name}.png", dpi=120)
            plt.close(fig)
