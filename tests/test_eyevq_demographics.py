from __future__ import annotations

import pytest

from eyemae.eyevq.downstream.demographics import (
    collect_subject_demographics,
    encode_subject_demographics,
    fit_demographic_spec,
    parse_sex,
)


def _row(subject: str, age: int, education: str, sex: str) -> dict[str, str]:
    return {
        "ml_subject_id": subject,
        "identity_age": str(age),
        "identity_education": education,
        "source_stem": f"cohort_{subject}_{age}_{sex}_{education}_trial",
    }


def test_sex_is_parsed_from_field_before_education() -> None:
    assert parse_sex("abc_70_F_CZ_trial") == "F"
    assert parse_sex("abc_75_M_BK_trial") == "M"
    assert parse_sex("missing") == "UNKNOWN"


def test_age_normalization_is_fit_on_train_fold_only() -> None:
    train = [_row("a", 60, "CZ", "F"), _row("b", 80, "BK", "M")]
    validation = [_row("c", 100, "XX", "F")]
    spec = fit_demographic_spec(train)
    assert spec["age_mean"] == 70.0
    encoded = encode_subject_demographics(validation, spec)["c"]
    assert encoded[0].item() > 0.0
    assert spec["fit_subjects"] == 2


def test_raw_age_encoding_keeps_years_and_records_its_semantics() -> None:
    train = [_row("a", 60, "CZ", "F"), _row("b", 80, "BK", "M")]
    validation = [_row("c", 100, "XX", "F")]
    spec = fit_demographic_spec(train, age_encoding="raw")
    encoded = encode_subject_demographics(validation, spec)["c"]
    assert spec["age_encoding"] == "raw"
    assert spec["feature_names"][0] == "age_raw"
    assert encoded[0].item() == 100.0
    assert encoded[1].item() == 0.0


def test_inconsistent_subject_metadata_is_rejected() -> None:
    rows = [_row("a", 60, "CZ", "F"), _row("a", 61, "CZ", "F")]
    with pytest.raises(ValueError, match="inconsistent demographic"):
        collect_subject_demographics(rows)
