import pytest

from eyemae.eyevq.downstream.cache_trial_cls import (
    QC_NAMES,
    trial_qc_from_row,
)


def test_trial_qc_uses_both_eyes_and_frame_denominator():
    row = {
        "left_final_keep": "True",
        "right_final_keep": "False",
        "left_blink_points": "20",
        "right_blink_points": "10",
        "left_missing_points": "40",
        "right_missing_points": "0",
        "frame_length": "100",
        "num_patches_20ms": "5",
    }
    values = trial_qc_from_row(row, valid_eye_token_fraction=0.45)
    assert len(values) == len(QC_NAMES)
    assert values == pytest.approx([1.0, 0.15, 0.20, 5.0, 0.45])


def test_trial_qc_marks_two_usable_eyes_as_not_one_eye():
    row = {
        "left_final_keep": "1",
        "right_final_keep": "yes",
        "frame_length": "0",
    }
    values = trial_qc_from_row(row, valid_eye_token_fraction=1.0)
    assert values == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])
