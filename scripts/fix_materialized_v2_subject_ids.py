from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.build_eyemae_fast_dataset_v2 import (  # noqa: E402
    DOWNSTREAM_INDEX_FIELDNAMES,
    IDENTITY_EDUCATION_OVERRIDES,
    PRETRAIN_INDEX_FIELDNAMES,
    SPLITS,
    parse_identity,
    summarize_rows,
    write_subjects,
)
from eyemae.build_fast_packed_dataset import _write_csv, _write_json  # noqa: E402


DEFAULT_ROOT = Path("/mnt/disk_sde/data-260606/extracted/eyemae_fast_dataset_v2")
DOWNSTREAM_TASKS = (
    "ad_binary",
    "detox_binary",
    "epilepsy_binary",
    "mci_binary",
    "mci_matched_binary",
    "migraine_binary",
    "pd_binary",
    "pd_related_5class",
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    _write_csv(tmp, rows, fieldnames)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    _write_json(tmp, payload)
    tmp.replace(path)


def fixed_identity(row: dict[str, str]):
    stem = row.get("source_stem", "").strip()
    if not stem:
        rel = row.get("relative_source_path", "").strip()
        stem = Path(rel).stem
    return parse_identity({"source_stem": stem, "subject": row.get("subject", "")})


def fix_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], Counter[str]]:
    changed = Counter()
    fixed: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        identity = fixed_identity(row)
        old_subject = row.get("ml_subject_id", "")
        if old_subject != identity.key:
            changed["ml_subject_id_changed"] += 1
        row["ml_subject_id"] = identity.key
        if "subject" in row:
            row["subject"] = identity.name
        if "raw_identity" in row:
            row["raw_identity"] = identity.key
        if "identity_name" in row:
            row["identity_name"] = identity.name
        if "identity_age" in row:
            row["identity_age"] = identity.age
        if "identity_education" in row:
            row["identity_education"] = identity.education
        fixed.append(row)
    return fixed, changed


def update_csv(path: Path, *, dry_run: bool) -> tuple[list[dict[str, str]], list[str], Counter[str]]:
    rows, fieldnames = read_csv(path)
    fixed, changed = fix_rows(rows)
    if not dry_run:
        atomic_write_csv(path, fixed, fieldnames)
    return fixed, fieldnames, changed


def update_manifest_identity_key(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identity_key"] = (
        "name|age|education_code parsed from source_stem; education_code is the "
        "field two positions before age, with configured metadata-inconsistent "
        "subjects kept as edu:MIXED to preserve their intended subject identity"
    )
    if not dry_run:
        atomic_write_json(path, payload)


def update_summary(path: Path, updates: dict[str, Any], *, dry_run: bool) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(updates)
    if not dry_run:
        atomic_write_json(path, payload)


def update_readme(root: Path, root_summary: dict[str, Any], *, dry_run: bool) -> None:
    pretrain = root_summary["pretrain"]
    downstream = root_summary["downstream"]
    lines = [
        "# EyeMAE Fast Dataset v2",
        "",
        "This is the current frozen fast dataset used by the new-data EyeMAE",
        "pretraining and downstream fine-tuning runs.",
        "",
        "Use this materialized directory as the source of truth:",
        "",
        "```text",
        str(root),
        "```",
        "",
        "The dataset is already packed for fast training. The data arrays were not",
        "rewritten for the subject-id fix; only CSV/JSON metadata was updated.",
        "",
        "## Source",
        "",
        "Built from:",
        "",
        "```text",
        root_summary.get("source_root", ""),
        "```",
        "",
        "Machine-readable build summary:",
        "",
        "```text",
        "v2_build_summary.json",
        "```",
        "",
        "## Subject Identity",
        "",
        "`ml_subject_id` uses the corrected identity key:",
        "",
        "```text",
        "name|age:<age>|edu:<education_code>",
        "```",
        "",
        "The education code is parsed from the filename field two positions before age",
        "such as `XX/CZ/GZ/ZZ/DZ/BK/SS/BS/WM`. Known metadata-inconsistent subjects",
        "are intentionally kept merged:",
        "",
        "```text",
        "FangDeXiu|age:77|edu:MIXED",
        "LuXingQiong|age:61|edu:MIXED",
        "雷妮莎|age:23|edu:MIXED",
        "```",
        "",
        "## Format",
        "",
        "This dataset is not stored as one trial per `.npz`. It is stored as packed",
        "memory-mapped shards plus CSV indexes.",
        "",
        "Each packed dataset root contains:",
        "",
        "```text",
        "shards/shard_xxxxxx/X_data.npy",
        "shards/shard_xxxxxx/y_frame.npy",
        "shards/shard_xxxxxx/X_offsets.npy",
        "shards/shard_xxxxxx/X_lengths.npy",
        "shards/shard_xxxxxx/trial_index.csv",
        "```",
        "",
        "Each CSV row points to one trial through:",
        "",
        "```text",
        "shard_id, local_trial_index, frame_offset, frame_length",
        "```",
        "",
        "`X_data` has 10 columns:",
        "",
        "```text",
        "left_x, left_y, left_area,",
        "right_x, right_y, right_area,",
        "stim_x, stim_y, stim_on, fix_on",
        "```",
        "",
        "`y_frame` has 2 columns:",
        "",
        "```text",
        "left_qc_label, right_qc_label",
        "```",
        "",
        "QC label mapping:",
        "",
        "```text",
        "0 = valid",
        "1 = blink",
        "2 = missing",
        "```",
        "",
        "## Labels",
        "",
        "For binary tasks:",
        "",
        "```text",
        "health_label = 0: healthy/control",
        "health_label = 1: disease/patient",
        "```",
        "",
        "For `pd_related_5class`, the training class is:",
        "",
        "```text",
        "0: control",
        "1: 帕金森病",
        "2: 震颤",
        "3: 特发性震颤",
        "4: 运动障碍",
        "```",
        "",
        "The CSV stores patient subtype in `pd_disease_label`:",
        "",
        "```text",
        "0: 帕金森病",
        "1: 震颤",
        "2: 特发性震颤",
        "3: 运动障碍",
        "-1: control or non-PD task",
        "```",
        "",
        "## Cleaning Rules Reflected In This Dataset",
        "",
        "- AD: `AD/匹配后/实验组` was excluded because it duplicates",
        "  `AD/AD组/患病`; `AD/匹配后/对照组/GaoLianYing` was excluded, so",
        "  GaoLianYing remains an AD patient only.",
        "- PD: matched experimental rows were excluded. PD matched controls were",
        "  deduplicated before creating the PD downstream tasks: if the same corrected",
        "  healthy-control identity appears in multiple PD matched-control folders,",
        "  only one canonical matched-control source is retained.",
        "- MCI original: `mci_binary` uses only the original MCI experiment/control",
        "  folders and excludes `MCI/匹配后`.",
        "- MCI matched: `mci_matched_binary` is a separate matched task. Its source",
        "  matched labels are reversed in this dataset so that `health_label=0` is",
        "  control and `health_label=1` is patient.",
        "- Epilepsy, migraine, and detox keep the v1 identity-level train/validation/test",
        "  split policy.",
        "",
        "## Pretrain Split",
        "",
        "| split | trials | subjects | frames |",
        "| --- | ---: | ---: | ---: |",
    ]
    total_rows = 0
    total_frames = 0
    for split in SPLITS:
        info = pretrain["splits"][split]
        total_rows += int(info["rows"])
        total_frames += int(info["frames"])
        lines.append(f"| {split} | {info['rows']:,} | {info['subjects']:,} | {info['frames']:,} |")
    lines.append(
        f"| total | {total_rows:,} | {pretrain.get('selected_identities', ''):,} | {total_frames:,} |"
    )
    lines.extend(
        [
            "",
            "Pretraining split policy:",
            "",
            "```text",
            "identity-held-out random split, 90/5/5",
            "```",
            "",
            "Subject overlap across train/validation/test is zero according to the stored",
            "build summary.",
            "",
            "## Downstream Tasks",
            "",
            "| task | trials | subjects | split policy |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for task in sorted(downstream):
        info = downstream[task]
        lines.append(
            f"| `{task}` | {info.get('selected_rows', 0):,} | "
            f"{info.get('selected_subjects', 0):,} | {info.get('split_policy', '')} |"
        )
    lines.extend(
        [
            "",
            "All downstream tasks have zero stored `ml_subject_id` overlap across",
            "train/validation/test.",
            "",
            "## Downstream Split Sizes",
            "",
            "| task | split | trials | subjects | subject labels |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for task in sorted(downstream):
        for split in SPLITS:
            info = downstream[task]["splits"][split]
            if task == "pd_related_5class":
                labels = info.get("class_subject_counts", {})
            else:
                labels = info.get("health_label_subject_counts", {})
            label_text = ", ".join(f"{k}:{v}" for k, v in sorted(labels.items(), key=lambda item: item[0]))
            lines.append(f"| `{task}` | {split} | {info['rows']:,} | {info['subjects']:,} | {label_text} |")
    lines.extend(
        [
            "",
            "## Use In Training",
            "",
            "Point pretraining configs at:",
            "",
            "```text",
            str(root / "pretrain" / "pretrain"),
            "```",
            "",
            "Point downstream configs at one of:",
            "",
            "```text",
            str(root / "finetune" / "<task>"),
            "```",
            "",
            "For reproducible experiments, consume the stored CSV indexes and shards in this",
            "directory. Historical audit documents in the repository explain why the cleaning",
            "rules were chosen, but this README and `v2_build_summary.json` define the",
            "current dataset version.",
        ]
    )
    if not dry_run:
        (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_pretrain(root: Path, root_summary: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    split_rows: dict[str, list[dict[str, str]]] = {}
    total_changed = Counter()
    for split in SPLITS:
        path = root / "pretrain" / "pretrain" / f"pretrain_{split}.csv"
        rows, _fields, changed = update_csv(path, dry_run=dry_run)
        split_rows[split] = rows
        total_changed.update(changed)
    all_rows, _fields, changed = update_csv(root / "pretrain" / "pretrain" / "pretrain_all_unique.csv", dry_run=dry_run)
    total_changed.update(changed)
    trials_rows, _fields, changed = update_csv(root / "pretrain" / "trials.csv", dry_run=dry_run)
    total_changed.update(changed)
    for shard_index in sorted((root / "pretrain" / "shards").glob("shard_*/trial_index.csv")):
        _rows, _fields, changed = update_csv(shard_index, dry_run=dry_run)
        total_changed.update(changed)
    if not dry_run:
        write_subjects(root / "pretrain" / "subjects.csv", trials_rows)

    summary = summarize_rows(split_rows, label_type="unlabeled")
    old = dict(root_summary.get("pretrain", {}))
    old.update(summary)
    old["selected_rows"] = len(all_rows)
    old["selected_identities"] = len({row["ml_subject_id"] for row in all_rows})
    old["subject_id_fix"] = {
        "status": "applied",
        "ml_subject_id_changed_rows": total_changed.get("ml_subject_id_changed", 0),
        "identity_key": "name|age|education_code with FangDeXiu/LuXingQiong MIXED overrides",
    }
    if not dry_run:
        update_summary(root / "pretrain" / "audit_summary.json", old, dry_run=False)
        update_summary(root / "pretrain" / "pretrain" / "pretrain_split_summary.json", old, dry_run=False)
        update_manifest_identity_key(root / "pretrain" / "dataset_manifest.json", dry_run=False)
    return old


def update_downstream(root: Path, root_summary: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    for task in DOWNSTREAM_TASKS:
        task_root = root / "finetune" / task
        split_rows: dict[str, list[dict[str, str]]] = {}
        total_changed = Counter()
        for split in SPLITS:
            rows, _fields, changed = update_csv(task_root / f"{split}.csv", dry_run=dry_run)
            split_rows[split] = rows
            total_changed.update(changed)
        trials_rows, _fields, changed = update_csv(task_root / "trials.csv", dry_run=dry_run)
        total_changed.update(changed)
        for shard_index in sorted((task_root / "shards").glob("shard_*/trial_index.csv")):
            _rows, _fields, changed = update_csv(shard_index, dry_run=dry_run)
            total_changed.update(changed)
        if not dry_run:
            write_subjects(task_root / "subjects.csv", trials_rows)

        label_type = "multiclass" if task == "pd_related_5class" else "binary"
        summary = summarize_rows(split_rows, label_type=label_type)
        old = dict(root_summary.get("downstream", {}).get(task, {}))
        old.update(summary)
        old["selected_rows"] = len(trials_rows)
        old["selected_subjects"] = len({row["ml_subject_id"] for row in trials_rows})
        old["subject_id_fix"] = {
            "status": "applied",
            "ml_subject_id_changed_rows": total_changed.get("ml_subject_id_changed", 0),
            "identity_key": "name|age|education_code with FangDeXiu/LuXingQiong MIXED overrides",
        }
        if not dry_run:
            update_summary(task_root / "split_summary.json", old, dry_run=False)
            update_summary(task_root / "audit_summary.json", old, dry_run=False)
            update_manifest_identity_key(task_root / "dataset_manifest.json", dry_run=False)
        updated[task] = old
    return updated


def validate_special_cases(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in ("pd_related_5class", "pd_binary"):
        path = root / "finetune" / task / "trials.csv"
        counts = Counter()
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["identity_name"] in {"FangDeXiu", "LuXingQiong"}:
                    counts[(row["identity_name"], row["ml_subject_id"], row["split"], row["health_label"], row["pd_disease_label"])] += 1
        result[task] = {str(key): value for key, value in sorted(counts.items())}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root
    root_summary_path = root / "v2_build_summary.json"
    root_summary = json.loads(root_summary_path.read_text(encoding="utf-8"))
    pretrain = update_pretrain(root, root_summary, dry_run=args.dry_run)
    downstream = update_downstream(root, root_summary, dry_run=args.dry_run)

    root_summary["pretrain"] = pretrain
    root_summary["downstream"] = downstream
    root_summary["subject_id_fix"] = {
        "status": "applied" if not args.dry_run else "dry_run",
        "identity_key": "name|age|education_code parsed from source_stem",
        "education_overrides": {
            f"{name}|age:{age}": education
            for (name, age), education in sorted(IDENTITY_EDUCATION_OVERRIDES.items())
        },
        "npy_array_shards_rewritten": False,
        "shard_trial_index_csv_rewritten": not args.dry_run,
        "csv_and_json_metadata_rewritten": not args.dry_run,
    }
    if not args.dry_run:
        atomic_write_json(root_summary_path, root_summary)
        update_readme(root, root_summary, dry_run=False)

    validation = {
        "pretrain_subjects": pretrain.get("selected_identities"),
        "pretrain_overlap": pretrain.get("subject_overlap_counts"),
        "downstream_subjects": {
            task: {
                "subjects": summary.get("selected_subjects"),
                "overlap": summary.get("subject_overlap_counts"),
                "no_subject_overlap": summary.get("no_subject_overlap"),
            }
            for task, summary in sorted(downstream.items())
        },
        "special_cases": validate_special_cases(root) if not args.dry_run else {},
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
