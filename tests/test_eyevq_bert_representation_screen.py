from pathlib import Path

import torch

from eyemae.eyevq.downstream.screen_bert_representation import subject_features


def test_subject_features_normalize_then_average_tasks(tmp_path: Path):
    features = []
    labels = []
    tasks = []
    subjects = []
    for subject, label, offset in (("a", 0, 0.0), ("b", 1, 3.0)):
        for task in range(4):
            for repeat in range(2):
                features.append(torch.tensor([offset + task, repeat + 1.0, -1.0]))
                labels.append(label)
                tasks.append(task)
                subjects.append(subject)
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "selection_source": "train_only_subject_folds",
            "bert_sha256": "abc",
            "splits": {"train": {
                "features": torch.stack(features),
                "labels": torch.tensor(labels),
                "task_ids": torch.tensor(tasks),
                "subject_keys": subjects,
            }},
        },
        path,
    )
    x, y, ordered, sha = subject_features(path)
    assert x.shape == (2, 3)
    assert y.tolist() == [0, 1]
    assert ordered == ["a", "b"]
    assert sha == "abc"


def test_subject_features_rejects_missing_task(tmp_path: Path):
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "selection_source": "train_only_subject_folds",
            "bert_sha256": "abc",
            "splits": {"train": {
                "features": torch.randn(3, 4),
                "labels": torch.zeros(3, dtype=torch.long),
                "task_ids": torch.tensor([0, 1, 2]),
                "subject_keys": ["a", "a", "a"],
            }},
        },
        path,
    )
    try:
        subject_features(path)
    except ValueError as error:
        assert "all four tasks" in str(error)
    else:
        raise AssertionError("missing task was accepted")
