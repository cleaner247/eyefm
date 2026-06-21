from __future__ import annotations

import argparse
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml


EPOCH_RE = re.compile(
    r"epoch=(\d+) train_loss=([0-9.eE+-]+|nan) "
    r"val_subject_metric=([0-9.eE+-]+|nan) val_subject_f1=([0-9.eE+-]+|nan)"
)
MODES = ["scratch", "linear_probe", "partial", "full"]
MODE_COLUMNS = {
    "scratch": "scratch",
    "linear_probe": "linear",
    "partial": "partial",
    "full": "full",
}


def read_config_list(path: Path) -> list[Path]:
    configs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        configs.append(Path(line))
    return configs


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return payload


def task_label(cfg: dict[str, Any]) -> str:
    task = str(cfg.get("downstream", {}).get("task_name") or cfg.get("label", {}).get("task_name"))
    output_dir = str(cfg.get("experiment", {}).get("output_dir", ""))
    parts = Path(output_dir).parts
    if len(parts) >= 2 and "_random_seed" in parts[-2]:
        return parts[-2]
    return task


def fmt_float(raw: str) -> str:
    value = float(raw)
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def parse_log(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match is None:
            continue
        epoch = int(match.group(1))
        values[epoch] = " / ".join(fmt_float(item) for item in match.groups()[1:])
    return values


def build_tables(config_list_file: Path, log_dir: Path) -> str:
    grouped: "OrderedDict[str, dict[str, dict[int, str]]]" = OrderedDict()
    for cfg_path in read_config_list(config_list_file):
        cfg = read_yaml(cfg_path)
        mode = str(cfg.get("downstream", {}).get("mode") or cfg.get("finetune", {}).get("mode"))
        if mode not in MODES:
            continue
        label = task_label(cfg)
        grouped.setdefault(label, {item: {} for item in MODES})
        grouped[label][mode] = parse_log(log_dir / f"{cfg_path.stem}.log")

    lines = [
        "Downstream epoch dynamics:",
        "",
        "Cell format: `train_loss / val_subject_metric / val_subject_f1`. Empty or",
        "not-yet-run cells are `-`. For binary tasks, `val_subject_metric` is AUROC; for",
        "the PD 5-class task, it is macro AUROC OVR.",
        "",
    ]
    for label, by_mode in grouped.items():
        if not any(by_mode[mode] for mode in MODES):
            continue
        max_epoch = max((max(values) for values in by_mode.values() if values), default=-1)
        lines.extend(
            [
                f"`{label}`",
                "",
                "| epoch | scratch | linear | partial | full |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for epoch in range(max_epoch + 1):
            cells = [by_mode[mode].get(epoch, "-") for mode in MODES]
            lines.append(f"| {epoch} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n\n"


def update_doc(doc_path: Path, replacement: str) -> None:
    text = doc_path.read_text(encoding="utf-8")
    start_marker = "Downstream epoch dynamics:\n"
    end_marker = "Downstream run guards and open gates:\n"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    updated = text[:start] + replacement + text[end:]
    doc_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list-file", default="configs/downstream/queue_pd_random_seed20260620_fast.txt")
    parser.add_argument("--log-dir", default="outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast")
    parser.add_argument("--doc", default="docs/newdata_v3_training_log.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    replacement = build_tables(Path(args.config_list_file), Path(args.log_dir))
    if args.dry_run:
        print(replacement)
        return
    update_doc(Path(args.doc), replacement)


if __name__ == "__main__":
    main()
