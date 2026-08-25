"""Downstream classification for EyeVQ encoders."""

from .mil_model import EyeVQSubjectMIL
from .model import EyeVQForClassification

__all__ = ["EyeVQForClassification", "EyeVQSubjectMIL"]
