"""Fold-safe subject demographic features for MCI downstream evaluation."""

from __future__ import annotations

import math
import re
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


SEX_CATEGORIES = ("F", "M", "UNKNOWN")
EDUCATION_CATEGORIES = (
    "BK", "BS", "CZ", "DZ", "GZ", "SS", "WM", "XX", "ZZ", "MIXED", "UNKNOWN"
)
AGE_ENCODINGS = ("zscore", "raw")
_SEX_PATTERN = re.compile(
    r"(?:^|_)(M|F)_(?:XX|CZ|GZ|ZZ|DZ|BK|SS|BS|WM|MIXED)(?:_|$)",
    re.IGNORECASE,
)


def _clean_token(value: Any, *, unknown: str = "UNKNOWN") -> str:
    text = str(value).strip().upper()
    return text if text and text not in {"NAN", "NONE"} else unknown


def parse_sex(source_stem: Any) -> str:
    """Parse the M/F metadata field immediately preceding education."""
    match = _SEX_PATTERN.search(str(source_stem))
    return match.group(1).upper() if match else "UNKNOWN"


def collect_subject_demographics(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collect one consistent age/education/sex record per subject."""
    observed: dict[str, dict[str, set[Any]]] = defaultdict(
        lambda: {"age": set(), "education": set(), "sex": set()}
    )
    for row in rows:
        subject = str(row["ml_subject_id"])
        try:
            age = float(row.get("identity_age", "nan"))
        except (TypeError, ValueError):
            age = math.nan
        if math.isfinite(age):
            observed[subject]["age"].add(age)
        observed[subject]["education"].add(
            _clean_token(row.get("identity_education", "UNKNOWN"))
        )
        observed[subject]["sex"].add(parse_sex(row.get("source_stem", "")))

    result: dict[str, dict[str, Any]] = {}
    for subject, fields in observed.items():
        conflicts = {
            name: sorted(values)
            for name, values in fields.items()
            if len(values) > 1
        }
        if conflicts:
            raise ValueError(
                f"Subject {subject} has inconsistent demographic metadata: {conflicts}"
            )
        result[subject] = {
            "age": next(iter(fields["age"]), math.nan),
            "education": next(iter(fields["education"]), "UNKNOWN"),
            "sex": next(iter(fields["sex"]), "UNKNOWN"),
        }
    return result


def read_demographic_index(
    path: str | Path, *, subjects: set[str] | None = None
) -> list[dict[str, Any]]:
    """Read metadata columns that the compact packed-index loader omits."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "ml_subject_id", "identity_age", "identity_education", "source_stem"
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Demographic index {path} is missing columns: {sorted(missing)}")
    if subjects is not None:
        rows = [row for row in rows if str(row["ml_subject_id"]) in subjects]
        found = {str(row["ml_subject_id"]) for row in rows}
        if found != subjects:
            raise ValueError(
                f"Demographic index {path} misses subjects: {sorted(subjects - found)[:10]}"
            )
    return rows


def fit_demographic_spec(
    rows: Iterable[dict[str, Any]], *, age_encoding: str = "zscore"
) -> dict[str, Any]:
    """Fit fold-safe demographic metadata and the requested age encoding."""
    age_encoding = str(age_encoding)
    if age_encoding not in AGE_ENCODINGS:
        raise ValueError(f"age_encoding must be one of {AGE_ENCODINGS}")
    subjects = collect_subject_demographics(rows)
    ages = [float(item["age"]) for item in subjects.values() if math.isfinite(item["age"])]
    if not ages:
        raise ValueError("No finite training ages are available")
    age_mean = sum(ages) / len(ages)
    variance = sum((age - age_mean) ** 2 for age in ages) / max(len(ages) - 1, 1)
    age_std = max(math.sqrt(variance), 1.0)
    feature_names = (
        "age_z" if age_encoding == "zscore" else "age_raw",
        "age_missing",
        *(f"sex_{value}" for value in SEX_CATEGORIES),
        *(f"education_{value}" for value in EDUCATION_CATEGORIES),
    )
    return {
        "age_encoding": age_encoding,
        "age_mean": age_mean,
        "age_std": age_std,
        "sex_categories": list(SEX_CATEGORIES),
        "education_categories": list(EDUCATION_CATEGORIES),
        "feature_names": list(feature_names),
        "feature_dim": len(feature_names),
        "fit_subjects": len(subjects),
    }


def encode_subject_demographics(
    rows: Iterable[dict[str, Any]], spec: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Encode subjects with train-fold age statistics and fixed categories."""
    subjects = collect_subject_demographics(rows)
    sex_categories = tuple(spec["sex_categories"])
    education_categories = tuple(spec["education_categories"])
    age_mean = float(spec["age_mean"])
    age_std = float(spec["age_std"])
    # Checkpoints written before age_encoding was introduced used z-scored age.
    age_encoding = str(spec.get("age_encoding", "zscore"))
    if age_encoding not in AGE_ENCODINGS:
        raise ValueError(f"age_encoding must be one of {AGE_ENCODINGS}")
    encoded: dict[str, torch.Tensor] = {}
    for subject, item in subjects.items():
        age = float(item["age"])
        age_missing = not math.isfinite(age)
        if age_missing:
            encoded_age = 0.0
        elif age_encoding == "raw":
            encoded_age = age
        else:
            encoded_age = (age - age_mean) / age_std
        values = [encoded_age, float(age_missing)]

        sex = item["sex"] if item["sex"] in sex_categories else "UNKNOWN"
        values.extend(float(sex == category) for category in sex_categories)

        education = (
            item["education"]
            if item["education"] in education_categories
            else "UNKNOWN"
        )
        values.extend(
            float(education == category) for category in education_categories
        )
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() != int(spec["feature_dim"]):
            raise RuntimeError("Demographic feature dimension does not match its spec")
        encoded[subject] = tensor
    return encoded
