"""Content-addressed identities for reproducible EyeVQ artifacts.

Paths alone are not identities: a checkpoint or dataset can be replaced at the
same path.  This module records hashes of the resolved configuration and every
small/medium dependency that determines a training target.  Large code caches
are hashed once in a sidecar manifest and validated before reuse.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from eyemae.utils import to_serializable, write_json


ARTIFACT_IDENTITY_VERSION = 1
CACHE_FORMAT_VERSION = 4


def checkpoint_step(path: str | Path) -> int:
    match = re.fullmatch(r"ckpt_step0*(\d+)\.pt", Path(path).name)
    if not match:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(1))


def latest_step_checkpoint(directory: str | Path) -> Path | None:
    candidates = list(Path(directory).glob("ckpt_step*.pt"))
    return max(candidates, key=checkpoint_step) if candidates else None


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    return to_serializable(value)


def semantic_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime-only keys and return a deterministically ordered config."""
    return _semantic_value(cfg)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _semantic_value(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Identity dependency does not exist: {file_path}")
    return {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def dataset_dependency_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    train_cfg = cfg["train"]
    data_cfg = cfg.get("data", {})
    data_root_value = train_cfg.get("data_path", data_cfg.get("data_dir"))
    if not data_root_value:
        raise ValueError("Config has neither train.data_path nor data.data_dir")
    data_path = Path(str(data_root_value)).resolve()
    dependencies: dict[str, Path] = {}
    manifest = data_path / "dataset_manifest.json"
    if manifest.is_file():
        dependencies["dataset_manifest"] = manifest
    for key in ("train_index", "val_index", "test_index"):
        value = train_cfg.get(key, data_cfg.get(key))
        if value:
            dependencies[key] = data_path / str(value)
    area_stats = train_cfg.get("area_stats_path", data_cfg.get("area_stats_path"))
    if area_stats:
        dependencies["area_stats"] = Path(str(area_stats))
    return dependencies


def build_run_identity(
    stage: str,
    cfg: Mapping[str, Any],
    *,
    dependencies: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    dependency_paths = dataset_dependency_paths(cfg)
    dependency_paths.update(
        {str(name): Path(path) for name, path in (dependencies or {}).items()}
    )
    dependency_identities = {
        name: file_identity(path)
        for name, path in sorted(dependency_paths.items())
    }
    semantic = semantic_config(cfg)
    return {
        "identity_version": ARTIFACT_IDENTITY_VERSION,
        "stage": str(stage),
        "config_sha256": sha256_json(semantic),
        "dependencies": dependency_identities,
    }


def assert_run_identity(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    allow_mismatch: bool = False,
) -> None:
    actual = checkpoint.get("run_identity")
    if actual == expected:
        return
    if allow_mismatch:
        return
    if actual is None:
        detail = "checkpoint has no run_identity"
    else:
        changed = sorted(
            key for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        detail = "mismatched fields: " + ", ".join(changed)
    raise ValueError(
        "Unsafe checkpoint resume rejected (" + detail + "). Use an explicit "
        "allow-resume-identity-mismatch option only for a deliberate schedule "
        "extension or transfer run."
    )


def cache_contract(cfg: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    """Return only fields that determine offline code-ID labels."""
    train_cfg = cfg["train"]
    return {
        "split": str(split),
        "patch": semantic_config(cfg.get("patch", {})),
        "vq": semantic_config(cfg.get("vq", {})),
        "area": semantic_config(cfg.get("area", {})),
        "max_patches": int(cfg.get("bert", {}).get("max_patches", 128)),
        "require_any_eye_keep": bool(train_cfg.get("require_any_eye_keep", True)),
        "data_path": str(Path(str(train_cfg["data_path"])).resolve()),
        "train_index": str(train_cfg.get("train_index", "")),
        "val_index": str(train_cfg.get("val_index", "")),
        "area_stats_path": str(Path(str(train_cfg["area_stats_path"])).resolve()),
    }


def cache_manifest_path(cache_path: str | Path) -> Path:
    cache = Path(cache_path)
    return cache.with_name(cache.name + ".manifest.json")


def write_cache_manifest(
    cache_path: str | Path,
    *,
    tokenizer_checkpoint: str | Path,
    contract_sha256: str,
    num_trials: int,
    legacy_payload_attested: bool = False,
) -> dict[str, Any]:
    cache = Path(cache_path).resolve()
    tokenizer = Path(tokenizer_checkpoint).resolve()
    payload = {
        "identity_version": ARTIFACT_IDENTITY_VERSION,
        "cache_format_version": (
            3 if legacy_payload_attested else CACHE_FORMAT_VERSION
        ),
        "cache": file_identity(cache),
        "tokenizer": file_identity(tokenizer),
        "cache_contract_sha256": str(contract_sha256),
        "num_trials": int(num_trials),
        "legacy_payload_attested": bool(legacy_payload_attested),
    }
    write_json(cache_manifest_path(cache), payload)
    return payload


def validate_cache_identity(
    cache_path: str | Path,
    *,
    tokenizer_checkpoint: str | Path,
    contract_sha256: str,
) -> dict[str, Any]:
    cache = Path(cache_path).resolve()
    tokenizer = Path(tokenizer_checkpoint).resolve()
    manifest_file = cache_manifest_path(cache)
    if not manifest_file.is_file():
        raise ValueError(f"Code-ID cache manifest is missing: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if int(manifest.get("identity_version", -1)) != ARTIFACT_IDENTITY_VERSION:
        raise ValueError("Unsupported code-ID cache identity version")
    manifest_format = int(manifest.get("cache_format_version", -1))
    if manifest_format not in {3, CACHE_FORMAT_VERSION}:
        raise ValueError("Unsupported code-ID cache format version")
    if manifest_format == 3 and manifest.get("legacy_payload_attested") is not True:
        raise ValueError("Legacy cache requires an explicit content-hash attestation")
    expected_cache = file_identity(cache)
    expected_tokenizer = file_identity(tokenizer)
    if manifest.get("cache") != expected_cache:
        raise ValueError("Code-ID cache content SHA256/size/path does not match its manifest")
    if manifest.get("tokenizer") != expected_tokenizer:
        raise ValueError("Code-ID cache tokenizer SHA256/size/path does not match")
    if manifest.get("cache_contract_sha256") != str(contract_sha256):
        raise ValueError("Code-ID cache preprocessing/FSQ/data contract does not match")
    with np.load(cache, allow_pickle=True) as payload:
        payload_format = int(payload["format_version"])
        if payload_format != manifest_format:
            raise ValueError("Code-ID cache payload format does not match its manifest")
        if payload_format == CACHE_FORMAT_VERSION:
            if str(payload["tokenizer_sha256"].item()) != expected_tokenizer["sha256"]:
                raise ValueError("Embedded tokenizer SHA256 does not match the current checkpoint")
            if str(payload["cache_contract_sha256"].item()) != str(contract_sha256):
                raise ValueError("Embedded cache contract does not match the requested contract")
        if len(payload["gids"]) != int(manifest["num_trials"]):
            raise ValueError("Code-ID cache trial count does not match its manifest")
    return manifest


def checkpoint_has_identity(
    checkpoint_path: str | Path,
    expected: Mapping[str, Any],
    *,
    required_step: int | None = None,
) -> bool:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert_run_identity(checkpoint, expected)
        if required_step is not None and int(checkpoint.get("step", -1)) != required_step:
            return False
        return True
    except (KeyError, OSError, RuntimeError, ValueError, TypeError, AttributeError,
            EOFError, pickle.UnpicklingError):
        return False
