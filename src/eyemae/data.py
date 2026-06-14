from __future__ import annotations

import logging
import re
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


def make_trial_dataset(cfg: dict[str, Any], split_file: str | Path, *, max_trials: int | None = None) -> TrialDataset:
    return TrialDataset(cfg["data"]["data_dir"], split_file, cfg, max_trials=max_trials)


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
        "trial_id": trial_id,
        "path": paths,
    }
