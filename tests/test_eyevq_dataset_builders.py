from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_view_collisions_use_per_trial_fallback(tmp_path: Path) -> None:
    module = load_script("build_eyemae_fast_dataset_v5_packed.py")
    module.V5_ROOT = tmp_path
    module.VIEWS = ("view_a", "view_b")
    module.SPLITS = ("train",)
    fieldnames = ["source_stem", "original_trial_index", "packed_trial_index"]
    for view, packed_index in (("view_a", 1), ("view_b", 2)):
        directory = tmp_path / view
        directory.mkdir()
        with (directory / "manifest_train.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "source_stem": "same_trial",
                    "original_trial_index": 7,
                    "packed_trial_index": packed_index,
                }
            )
        (directory / "train.npz").touch()
    assert "same_trial|7" not in module.build_view_map()


def test_v6_manifest_preserves_inherited_filter_provenance(tmp_path: Path) -> None:
    module = load_script("build_eyemae_fast_dataset_v6.py")
    manifest = tmp_path / "pretrain/dataset_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "signal_filter_policy": "guarded_75hz",
                "signal_filter_provenance_sha256": "abc",
            }
        ),
        encoding="utf-8",
    )
    assert module.rewrite_manifests(tmp_path) == 1
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "inherited_from_v5_guarded_75hz" in payload["signal_filter_policy"]
    assert payload["signal_filter_provenance_sha256"] == "abc"
    assert "no additional filtering" in payload["signal_policy"]
