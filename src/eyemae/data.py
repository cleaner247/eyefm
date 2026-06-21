from __future__ import annotations

import logging
import csv
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .patching import patchify_preprocessed_trial
from .preprocess import load_area_stats, preprocess_trial


LOGGER = logging.getLogger(__name__)

TASK_TO_ID = {
    "pro": 0,
    "prosaccade": 0,
    "正向": 0,
    "anti": 1,
    "antisaccade": 1,
    "反向": 1,
    "memory": 2,
    "memorysaccade": 2,
    "memsaccade": 2,
    "记忆": 2,
    "double": 3,
    "doublesaccade": 3,
    "二次": 3,
}
ID_TO_TASK = {0: "pro", 1: "anti", 2: "memory", 3: "double"}


def task_name_to_id(value: str) -> int:
    key = value.replace("_", "").replace("-", "").lower()
    if key not in TASK_TO_ID:
        raise ValueError(f"Unknown task name: {value}")
    return int(TASK_TO_ID[key])


def infer_task_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        key = part.replace("_", "").replace("-", "").lower()
        if key in TASK_TO_ID:
            return int(TASK_TO_ID[key])
    raise ValueError(f"Cannot infer task_id from path: {path}")


def infer_suffix_from_name(path: Path) -> str:
    match = re.search(r"_(D|L|R)(?:_|$)", path.stem)
    if match:
        return match.group(1)
    for part in reversed(path.parts):
        if part and part[-1] in {"D", "L", "R"}:
            return part[-1]
    return "D"


def infer_subject_from_path(path: Path, data_dir: Path) -> str:
    suffix = infer_suffix_from_name(path)
    parts = path.relative_to(data_dir).parts
    subject = ""
    if len(parts) >= 3:
        # Structured real data: disease/group/subject/task/file.npz.
        subject = parts[-3]
    elif len(parts) >= 2:
        subject = Path(parts[-2]).name
    if not subject:
        subject = path.parent.name
    if subject.endswith(("D", "L", "R")):
        return subject
    return f"{subject}{suffix}"


def infer_trial_id_from_path(path: Path) -> str:
    return path.stem


def get_split_subject_key(subject_id: str, group_by_base_subject_id: bool = True) -> str:
    if group_by_base_subject_id and len(subject_id) > 0 and subject_id[-1] in {"D", "L", "R"}:
        return subject_id[:-1]
    return subject_id


def parse_subject_eye_availability(subject_id: str) -> dict[str, Any]:
    if len(subject_id) == 0:
        raise ValueError("empty subject_id")
    suffix = subject_id[-1]
    if suffix == "D":
        return {"left_available": True, "right_available": True, "suffix": "D"}
    if suffix == "L":
        return {"left_available": True, "right_available": False, "suffix": "L"}
    if suffix == "R":
        return {"left_available": False, "right_available": True, "suffix": "R"}
    raise ValueError(f"Unknown subject suffix: {subject_id}")


def parse_eye_availability_suffix(value: str | None) -> dict[str, Any]:
    suffix = (value or "D").strip() or "D"
    suffix = suffix[-1] if suffix[-1] in {"D", "L", "R"} else "D"
    if suffix == "D":
        return {"left_available": True, "right_available": True, "suffix": "D"}
    if suffix == "L":
        return {"left_available": True, "right_available": False, "suffix": "L"}
    return {"left_available": False, "right_available": True, "suffix": "R"}


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _np_scalar_to_py(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(()).item()
    return value


def _read_optional_string(z: np.lib.npyio.NpzFile, key: str | None) -> str | None:
    if not key or key not in z.files:
        return None
    value = _np_scalar_to_py(z[key])
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _load_canonical_npz(path: Path, data_dir: Path, cfg: dict[str, Any], z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    keys = cfg["data"]["npz_keys"]
    eye = np.asarray(z[keys["eye"]], dtype=np.float32)
    task_key = keys.get("task_id", "task_id")
    if task_key in z.files:
        task_id = int(_np_scalar_to_py(z[task_key]))
    else:
        task_id = infer_task_from_path(path)
        LOGGER.warning("task_id missing in %s; inferred from path", path)
    fix_on = np.asarray(z[keys["fix_on"]], dtype=np.float32)
    stim = np.asarray(z[keys["stim"]], dtype=np.float32)
    subject_id = _read_optional_string(z, keys.get("subject_id"))
    if not subject_id:
        subject_id = infer_subject_from_path(path, data_dir)
        LOGGER.warning("subject_id missing in %s; inferred as %s", path, subject_id)
    trial_id = _read_optional_string(z, keys.get("trial_id"))
    if not trial_id:
        trial_id = infer_trial_id_from_path(path)
        LOGGER.warning("trial_id missing in %s; inferred as %s", path, trial_id)
    return {
        "eye": eye,
        "task_id": np.asarray(task_id, dtype=np.int64),
        "fix_on": fix_on,
        "stim": stim,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "path": str(path),
    }


def _load_cd_no_cond2_npz(path: Path, data_dir: Path, cfg: dict[str, Any], z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    keys = cfg["data"]["npz_keys"]
    gaze = np.asarray(z[keys.get("eye", "gaze")], dtype=np.float32)
    stimulus = np.asarray(z[keys.get("stimulus", "stimulus")], dtype=np.float32)
    gc = cfg["data"].get("gaze_columns", {})
    sc = cfg["data"].get("stimulus_columns", {})
    eye = np.stack(
        [
            gaze[:, int(gc.get("left_x", 0))],
            gaze[:, int(gc.get("left_y", 1))],
            gaze[:, int(gc.get("left_area", 2))],
            gaze[:, int(gc.get("left_label", 6))],
            gaze[:, int(gc.get("right_x", 3))],
            gaze[:, int(gc.get("right_y", 4))],
            gaze[:, int(gc.get("right_area", 5))],
            gaze[:, int(gc.get("right_label", 7))],
        ],
        axis=1,
    ).astype(np.float32)
    fix_on = stimulus[:, int(sc.get("fix_on", 3))].astype(np.float32)
    stim = np.stack(
        [
            stimulus[:, int(sc.get("stim_on", 2))],
            stimulus[:, int(sc.get("stim_x", 0))],
            stimulus[:, int(sc.get("stim_y", 1))],
        ],
        axis=1,
    ).astype(np.float32)
    task_id = infer_task_from_path(path)
    subject_id = infer_subject_from_path(path, data_dir)
    return {
        "eye": eye,
        "task_id": np.asarray(task_id, dtype=np.int64),
        "fix_on": fix_on,
        "stim": stim,
        "subject_id": subject_id,
        "trial_id": infer_trial_id_from_path(path),
        "path": str(path),
    }


def load_npz_trial(path: str | Path, data_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    p = Path(path)
    root = Path(data_dir)
    with np.load(p, allow_pickle=True) as z:
        schema = cfg["data"].get("npz_schema", "canonical")
        if schema == "canonical":
            trial = _load_canonical_npz(p, root, cfg, z)
        elif schema == "cd_no_cond2_gaze_stimulus":
            trial = _load_cd_no_cond2_npz(p, root, cfg, z)
        else:
            raise ValueError(f"Unknown data.npz_schema: {schema}")
    trial = apply_nan_policy(trial, cfg)
    validate_npz_trial(trial, cfg, path=p)
    return trial


def apply_nan_policy(trial: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["data"].get("nan_policy", "error") != "mark_missing":
        return trial
    eye = np.asarray(trial["eye"], dtype=np.float32).copy()
    missing_value = float(cfg["label"]["missing_value"])
    for offset in (0, 4):
        feature_cols = [offset, offset + 1, offset + 2]
        label_col = offset + 3
        feature_nan = np.isnan(eye[:, feature_cols]).any(axis=1)
        label_nan = np.isnan(eye[:, label_col])
        mark = feature_nan | label_nan
        if mark.any():
            eye[mark, label_col] = missing_value
            eye[mark, offset : offset + 3] = 0.0
    trial = dict(trial)
    trial["eye"] = eye
    fix_on = np.asarray(trial["fix_on"], dtype=np.float32).copy()
    if np.isnan(fix_on).any():
        fix_on[np.isnan(fix_on)] = 0.0
        trial["fix_on"] = fix_on
    stim = np.asarray(trial["stim"], dtype=np.float32).copy()
    if np.isnan(stim).any():
        bad = np.isnan(stim).any(axis=1)
        stim[bad] = 0.0
        trial["stim"] = stim
    return trial


def validate_npz_trial(trial: dict[str, Any], cfg: dict[str, Any], path: str | Path | None = None) -> None:
    prefix = f"{path}: " if path is not None else ""
    eye = np.asarray(trial["eye"])
    if eye.ndim != 2 or eye.shape[1] != 8:
        raise ValueError(f"{prefix}eye must have shape [T, 8], got {eye.shape}")
    t = eye.shape[0]
    fix_on = np.asarray(trial["fix_on"])
    stim = np.asarray(trial["stim"])
    if fix_on.shape != (t,):
        raise ValueError(f"{prefix}fix_on must have shape [{t}], got {fix_on.shape}")
    if stim.shape != (t, 3):
        raise ValueError(f"{prefix}stim must have shape [{t}, 3], got {stim.shape}")
    task_id = int(_np_scalar_to_py(trial["task_id"]))
    if task_id not in {0, 1, 2, 3}:
        raise ValueError(f"{prefix}task_id must be in {{0,1,2,3}}, got {task_id}")
    label_values = {
        int(cfg["label"]["nonblink_value"]),
        int(cfg["label"]["blink_value"]),
        int(cfg["label"]["missing_value"]),
    }
    left_labels = eye[:, 3]
    right_labels = eye[:, 7]
    observed_labels = set(np.unique(np.concatenate([left_labels, right_labels])).astype(int).tolist())
    if not observed_labels <= label_values:
        raise ValueError(f"{prefix}labels must be subset of {label_values}, got {sorted(observed_labels)}")
    for name, arr in (("eye", eye), ("fix_on", fix_on), ("stim", stim)):
        if np.isinf(arr).any():
            raise ValueError(f"{prefix}{name} contains inf")
        if np.isnan(arr).any():
            raise ValueError(f"{prefix}{name} contains NaN")
    if not isinstance(trial.get("subject_id"), str) or not trial["subject_id"]:
        raise ValueError(f"{prefix}subject_id must be a non-empty string")
    if not isinstance(trial.get("trial_id"), str) or not trial["trial_id"]:
        raise ValueError(f"{prefix}trial_id must be a non-empty string")


PACKED_REQUIRED_COLUMNS = {
    "global_trial_id",
    "shard_id",
    "local_trial_index",
    "frame_offset",
    "frame_length",
    "ml_subject_id",
    "task_id",
}


def _minimal_packed_row(row: dict[str, str]) -> dict[str, str]:
    keep = [
        "global_trial_id",
        "shard_id",
        "local_trial_index",
        "frame_offset",
        "frame_length",
        "num_patches_20ms",
        "ml_subject_id",
        "subject",
        "trial_id",
        "task_id",
        "source_suffix",
        "left_final_keep",
        "right_final_keep",
        "health_label",
        "pd_disease_label",
    ]
    return {key: row.get(key, "") for key in keep}


def read_packed_index(index_file: str | Path) -> list[dict[str, str]]:
    path = Path(index_file)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Packed index has no header: {path}")
        missing = sorted(PACKED_REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Packed index {path} missing columns: {missing}")
        rows = [_minimal_packed_row(row) for row in reader]
    if not rows:
        raise ValueError(f"Packed index is empty: {path}")
    return rows


class PackedTrialStore:
    def __init__(
        self,
        data_dir: str | Path,
        *,
        max_open_shards_per_worker: int = 16,
        validate_offsets: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.shards_dir = self.data_dir / "shards"
        self.max_open_shards = max(1, int(max_open_shards_per_worker))
        self.validate_offsets = bool(validate_offsets)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _open_shard(self, shard_id: str) -> dict[str, Any]:
        cached = self._cache.get(shard_id)
        if cached is not None:
            self._cache.move_to_end(shard_id)
            return cached
        shard_dir = self.shards_dir / shard_id
        payload = {
            "X": np.load(shard_dir / "X_data.npy", mmap_mode="r"),
            "Y": np.load(shard_dir / "y_frame.npy", mmap_mode="r"),
            "offsets": np.load(shard_dir / "X_offsets.npy", mmap_mode="r"),
            "lengths": np.load(shard_dir / "X_lengths.npy", mmap_mode="r"),
        }
        self._cache[shard_id] = payload
        if len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return payload

    def read_trial(self, row: dict[str, str]) -> dict[str, Any]:
        shard_id = row["shard_id"]
        local_idx = int(row["local_trial_index"])
        start = int(row["frame_offset"])
        length = int(row["frame_length"])
        end = start + length
        shard = self._open_shard(shard_id)
        if self.validate_offsets:
            expected_start = int(shard["offsets"][local_idx])
            expected_length = int(shard["lengths"][local_idx])
            if start != expected_start or length != expected_length:
                raise ValueError(
                    "Packed index offset mismatch for "
                    f"global_trial_id={row.get('global_trial_id')} shard_id={shard_id} "
                    f"local_trial_index={local_idx}: csv=({start},{length}) "
                    f"npy=({expected_start},{expected_length})"
                )
        x = np.asarray(shard["X"][start:end], dtype=np.float32)
        y = np.asarray(shard["Y"][start:end], dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != 10:
            raise ValueError(f"X_data slice must have shape [T,10], got {x.shape} for {row.get('global_trial_id')}")
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"y_frame slice must have shape [T,2], got {y.shape} for {row.get('global_trial_id')}")
        eye = np.stack(
            [
                x[:, 0],
                x[:, 1],
                x[:, 2],
                y[:, 0].astype(np.float32),
                x[:, 3],
                x[:, 4],
                x[:, 5],
                y[:, 1].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        stim = np.stack([x[:, 8], x[:, 6], x[:, 7]], axis=1).astype(np.float32)
        suffix = row.get("source_suffix") or row.get("subject") or row.get("ml_subject_id")
        availability = parse_eye_availability_suffix(suffix)
        return {
            "eye": eye,
            "task_id": np.asarray(int(row["task_id"]), dtype=np.int64),
            "fix_on": x[:, 9].astype(np.float32),
            "stim": stim,
            "subject_id": row["ml_subject_id"],
            "ml_subject_id": row["ml_subject_id"],
            "trial_id": row["global_trial_id"],
            "global_trial_id": row["global_trial_id"],
            "source_trial_id": row.get("trial_id", ""),
            "path": f'{row["shard_id"]}:{row["local_trial_index"]}',
            "source_suffix": availability["suffix"],
            "left_eye_available": _as_bool(row.get("left_final_keep"), availability["left_available"]),
            "right_eye_available": _as_bool(row.get("right_final_keep"), availability["right_available"]),
        }


class PackedPretrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        index_file: str | Path,
        cfg: dict[str, Any],
        *,
        area_stats: dict[str, Any] | None = None,
        max_trials: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.index_file = Path(index_file)
        self.rows = read_packed_index(self.index_file)
        if max_trials is not None:
            self.rows = self.rows[: int(max_trials)]
        self.area_stats = area_stats if area_stats is not None else load_area_stats(cfg["area"]["stats_path"])
        self.store = PackedTrialStore(
            self.data_dir,
            max_open_shards_per_worker=int(cfg["data"].get("max_open_shards_per_worker", 16)),
            validate_offsets=bool(cfg["data"].get("validate_offsets", True)),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def get_num_patches(self, index: int) -> int:
        row = self.rows[index]
        if row.get("num_patches_20ms"):
            return int(row["num_patches_20ms"])
        return int(row["frame_length"]) // int(self.cfg["patch"]["samples"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        trial = self.store.read_trial(row)
        validate_npz_trial(trial, self.cfg, path=trial["path"])
        processed = preprocess_trial(trial, self.cfg, self.area_stats)
        processed["ml_subject_id"] = trial["ml_subject_id"]
        processed["global_trial_id"] = trial["global_trial_id"]
        processed["source_suffix"] = trial.get("source_suffix", "")
        patched = patchify_preprocessed_trial(processed, self.cfg)
        if patched is None:
            raise ValueError(f"No valid patches in packed trial: {trial['path']}")
        patched["ml_subject_id"] = trial["ml_subject_id"]
        patched["global_trial_id"] = trial["global_trial_id"]
        patched["source_suffix"] = trial.get("source_suffix", "")
        return patched


def read_split_file(split_file: str | Path) -> list[str]:
    rows: list[str] = []
    for raw in Path(split_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


class TrialDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        cfg: dict[str, Any],
        *,
        area_stats: dict[str, Any] | None = None,
        max_trials: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.rel_paths = read_split_file(split_file)
        if max_trials is not None:
            self.rel_paths = self.rel_paths[: int(max_trials)]
        self.paths = [self.data_dir / rel for rel in self.rel_paths]
        self.area_stats = area_stats if area_stats is not None else load_area_stats(cfg["area"]["stats_path"])
        self._length_cache: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def get_num_patches(self, index: int) -> int:
        if index in self._length_cache:
            return self._length_cache[index]
        path = self.paths[index]
        schema = self.cfg["data"].get("npz_schema", "canonical")
        key = self.cfg["data"]["npz_keys"].get("eye", "eye")
        if schema == "cd_no_cond2_gaze_stimulus":
            key = self.cfg["data"]["npz_keys"].get("eye", "gaze")
        with np.load(path, allow_pickle=True) as z:
            t = int(z[key].shape[0])
        n = t // int(self.cfg["patch"]["samples"])
        self._length_cache[index] = n
        return n

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        trial = load_npz_trial(path, self.data_dir, self.cfg)
        processed = preprocess_trial(trial, self.cfg, self.area_stats)
        patched = patchify_preprocessed_trial(processed, self.cfg)
        if patched is None:
            raise ValueError(f"No valid patches in trial: {path}")
        return patched


def make_trial_dataset(cfg: dict[str, Any], split_file: str | Path, *, max_trials: int | None = None) -> Dataset:
    if cfg["data"].get("format") == "packed_mmap":
        return PackedPretrainDataset(cfg["data"]["data_dir"], split_file, cfg, max_trials=max_trials)
    return TrialDataset(cfg["data"]["data_dir"], split_file, cfg, max_trials=max_trials)


def audit_packed_pretrain_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["data"].get("format") != "packed_mmap":
        return {}
    data_dir = Path(cfg["data"]["data_dir"])
    for rel in (
        "dataset_manifest.json",
        "audit_summary.json",
        str(cfg.get("split", {}).get("split_summary", "pretrain/pretrain_split_summary.json")),
    ):
        path = data_dir / rel
        if not path.exists():
            raise ValueError(f"Required packed dataset audit file does not exist: {path}")
    split_files = {
        "train": Path(cfg["data"]["train_index"]),
        "validation": Path(cfg["data"]["val_index"]),
        "test": Path(cfg["data"]["test_index"]),
    }
    subject_sets: dict[str, set[str]] = {}
    global_ids: set[str] = set()
    max_patches = 0
    counts: dict[str, int] = {}
    for split, rel_path in split_files.items():
        rows = read_packed_index(data_dir / rel_path)
        counts[split] = len(rows)
        subjects = {row["ml_subject_id"] for row in rows}
        subject_sets[split] = subjects
        for row in rows:
            gid = row["global_trial_id"]
            if gid in global_ids:
                raise ValueError(f"Duplicate global_trial_id across pretrain splits: {gid}")
            global_ids.add(gid)
            task_id = int(row["task_id"])
            if task_id not in {0, 1, 2, 3}:
                raise ValueError(f"Invalid task_id={task_id} for global_trial_id={gid}")
            patches = int(row.get("num_patches_20ms") or (int(row["frame_length"]) // int(cfg["patch"]["samples"])))
            max_patches = max(max_patches, patches)
    overlaps = {
        "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "validation_test": len(subject_sets["validation"] & subject_sets["test"]),
    }
    if any(value != 0 for value in overlaps.values()):
        raise ValueError(f"Packed pretrain subject overlap is not zero: {overlaps}")
    configured_max = int(cfg["model"].get("max_patches", 0))
    if configured_max > 0 and max_patches > configured_max:
        raise ValueError(f"model.max_patches={configured_max} is smaller than required patches={max_patches}")
    return {"split_counts": counts, "subject_overlaps": overlaps, "max_required_patches": max_patches}


def collate_trials(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty batch")
    batch_size = len(items)
    nmax = max(int(item["content"].shape[0]) for item in items)
    patch = int(items[0]["content"].shape[2])
    content = torch.zeros(batch_size, nmax, 2, patch, 4, dtype=torch.float32)
    quality = torch.ones(batch_size, nmax, 2, patch, 1, dtype=torch.float32)
    stim = torch.zeros(batch_size, nmax, patch, 4, dtype=torch.float32)
    task_id = torch.zeros(batch_size, dtype=torch.long)
    pad_mask = torch.ones(batch_size, nmax, dtype=torch.bool)
    eye_nonmissing_frac = torch.zeros(batch_size, nmax, 2, dtype=torch.float32)
    eye_token_valid = torch.zeros(batch_size, nmax, 2, dtype=torch.bool)
    subject_id: list[str] = []
    trial_id: list[str] = []
    ml_subject_id: list[str] = []
    global_trial_id: list[str] = []
    source_suffix: list[str] = []
    paths: list[str] = []
    for b, item in enumerate(items):
        n = int(item["content"].shape[0])
        content[b, :n] = torch.as_tensor(item["content"], dtype=torch.float32)
        quality[b, :n] = torch.as_tensor(item["quality"], dtype=torch.float32)
        stim[b, :n] = torch.as_tensor(item["stim"], dtype=torch.float32)
        task_id[b] = int(item["task_id"])
        pad_mask[b, :n] = False
        eye_nonmissing_frac[b, :n] = torch.as_tensor(item["eye_nonmissing_frac"], dtype=torch.float32)
        eye_token_valid[b, :n] = torch.as_tensor(item["eye_token_valid"], dtype=torch.bool)
        subject_id.append(str(item["subject_id"]))
        trial_id.append(str(item["trial_id"]))
        ml_subject_id.append(str(item.get("ml_subject_id", item["subject_id"])))
        global_trial_id.append(str(item.get("global_trial_id", item["trial_id"])))
        source_suffix.append(str(item.get("source_suffix", "")))
        paths.append(str(item.get("path", "")))
    return {
        "content": content,
        "quality": quality,
        "stim": stim,
        "task_id": task_id,
        "pad_mask": pad_mask,
        "eye_nonmissing_frac": eye_nonmissing_frac,
        "eye_token_valid": eye_token_valid,
        "subject_id": subject_id,
        "ml_subject_id": ml_subject_id,
        "trial_id": trial_id,
        "global_trial_id": global_trial_id,
        "source_suffix": source_suffix,
        "path": paths,
    }
