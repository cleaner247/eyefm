from __future__ import annotations

import torch

from eyemae.model import EyeMAEModel


def test_sequence_mapping_indices() -> None:
    s = torch.full((1, 3, 2), 10.0)
    l = torch.full((1, 3, 2), 20.0)
    r = torch.full((1, 3, 2), 30.0)
    seq = EyeMAEModel.interleave_sequence(s, l, r)
    assert torch.equal(seq[:, 0::3], s)
    assert torch.equal(seq[:, 1::3], l)
    assert torch.equal(seq[:, 2::3], r)
    eye = EyeMAEModel.extract_eye_hidden(seq)
    assert torch.equal(eye[:, :, 0], l)
    assert torch.equal(eye[:, :, 1], r)
