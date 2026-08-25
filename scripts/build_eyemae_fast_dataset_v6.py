#!/usr/bin/env python3
"""Repack the inherited V5 guarded-75-Hz payload as V6.

The established V5 subject/trial split and identity tables are reused. Signal
payloads and frame labels are read from the disease-view NPZ files (with the
same-export per-trial NPZ fallback for pretraining-only trials). This builder
does not run an *additional* filter or recompute QC labels; it does not recover
the pre-V5 signal and must never be described as a raw/no-filter dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/mnt/disk_sde/hjf/eyemae_fast_dataset_v5")
OUTPUT_ROOT = Path("/mnt/disk_sde/hjf/eyemae_fast_dataset_v6")
BASE_BUILDER = REPO_ROOT / "scripts" / "build_eyemae_fast_dataset_v5_packed.py"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("v5_packer_for_v6", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def split_hashes(root: Path) -> dict[str, str]:
    names = {
        "train.csv", "validation.csv", "test.csv",
        "pretrain_train.csv", "pretrain_validation.csv", "pretrain_test.csv",
        "manifest_train.csv", "manifest_validation.csv", "manifest_test.csv",
    }
    result: dict[str, str] = {}
    for path in root.rglob("*.csv"):
        if path.name in names:
            result[str(path.relative_to(root))] = sha256(path)
    return result


def rewrite_manifests(output: Path) -> int:
    count = 0
    for manifest in output.rglob("dataset_manifest.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.update(
            {
                "dataset_version": "eyemae_fast_dataset_v6",
                "derived_from_dataset": str(SOURCE_ROOT),
                "source_root": str(SOURCE_ROOT),
                "payload_source": str(SOURCE_ROOT),
                "signal_policy": (
                    "bitwise repack of V5 guarded-75-Hz payload; "
                    "no additional filtering in the V6 builder"
                ),
                "signal_filter_policy": (
                    "inherited_from_v5_guarded_75hz_butterworth_lowpass; "
                    "blink_missing_and_saccade_protected"
                ),
                "frame_label_policy": "direct source NPZ gaze columns 6/7; no QC recomputation",
                "split_policy": "reuse established V5 subject/trial assignments",
            }
        )
        atomic_json(manifest, payload)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_ROOT}")
    for required in (SOURCE_ROOT, BASE_BUILDER):
        if not required.exists():
            raise FileNotFoundError(required)

    before_splits = split_hashes(SOURCE_ROOT)
    builder = load_builder()
    # Source identities/splits come from V5 packed metadata; payload lookup
    # continues to use V5 disease views and its same-export per-trial source.
    builder.V4_ROOT = SOURCE_ROOT
    builder.V5_ROOT = SOURCE_ROOT
    builder.V5_RAW = SOURCE_ROOT / "_raw_structured_tmp"
    builder.OUT_ROOT = OUTPUT_ROOT
    builder.UNPACKED_DIR = SOURCE_ROOT / "_unpacked_npy"
    builder._VIEW_MAP, builder._PER_TRIAL_MAP = builder.load_source_map()

    summaries = []
    for dataset in ("pretrain", *builder.TASKS):
        summaries.append(builder.build_dataset(dataset, args.workers))

    manifests = rewrite_manifests(OUTPUT_ROOT)
    after_splits = split_hashes(OUTPUT_ROOT)
    comparable = sorted(set(before_splits) & set(after_splits))
    split_mismatches = [name for name in comparable if before_splits[name] != after_splits[name]]
    expected_packed = {
        name: digest for name, digest in before_splits.items()
        if name.startswith("pretrain/") or name.startswith("finetune/")
    }
    missing_splits = sorted(set(expected_packed) - set(after_splits))

    summary = {
        "dataset_version": "eyemae_fast_dataset_v6",
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "policy": {
            "coordinates": (
                "bitwise V5 guarded-75-Hz payload; V6 applies no additional filter"
            ),
            "frame_labels": "read directly from source gaze[:,6:8]; no QC recomputation",
            "splits": "V5 subject/trial assignments preserved",
            "packed_invalid_values": "NaN in invalid frames replaced by zero only in packed tensors",
        },
        "rewritten_manifests": manifests,
        "datasets": summaries,
        "split_audit": {
            "compared_files": len(comparable),
            "mismatches": split_mismatches,
            "missing": missing_splits,
        },
        "passed": not split_mismatches and not missing_splits,
    }
    atomic_json(OUTPUT_ROOT / "v6_build_audit.json", summary)
    (OUTPUT_ROOT / "README_V6_CN.md").write_text(
        "# EyeMAE Fast Dataset V6\n\n"
        "V6 直接读取 `eyemae_fast_dataset_v5` 各疾病视图的 NPZ 及同批 per-trial NPZ，"
        "因此继承 V5 的保护式 75 Hz 处理结果，并沿用既有 subject/trial 划分。V6 构建"
        "阶段不再追加滤波、不重新运行 QC，也不改写"
        "VALID/BLINK/MISSING 标签。训练 packed 张量中仅将无效帧 NaN 置零，标签负责屏蔽。\n",
        encoding="utf-8",
    )
    if not summary["passed"]:
        raise RuntimeError("V6 split audit failed; see v6_build_audit.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
