"""EyeVQ-BERT for downstream classification.

CLS token → MLP → binary logit (or N-class logits).
Aligned with EyeMAE DownstreamClassifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eyemae.eyevq.pretrain.model import EyeVQBERT


class EyeVQForClassification(nn.Module):
    """BERT + classification head.

    For binary: out_dim=1 → BCEWithLogitsLoss (single logit).
    For multiclass: out_dim=num_classes → CrossEntropyLoss.
    """

    def __init__(self, bert: EyeVQBERT, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.bert = bert
        # Downstream classification calls ``bert.encode`` without masked-token
        # prediction.  These pretraining-only parameters are therefore outside
        # the downstream computation graph.  Freeze them explicitly so they do
        # not enter the optimizer or violate DDP's all-parameter reduction
        # invariant.
        self.bert.embed.mask_token.requires_grad_(False)
        self.bert.pred_head.requires_grad_(False)
        if self.bert.manual_feat_head is not None:
            self.bert.manual_feat_head.requires_grad_(False)
        if self.bert.span_length_embed is not None:
            self.bert.span_length_embed.requires_grad_(False)
        d_model = bert.d_model
        self.num_classes = num_classes
        # Binary → single logit; multiclass → num_classes logits
        out_dim = 1 if num_classes == 2 else num_classes
        hidden_dim = 256
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        stim_patches: torch.Tensor,
        eye_patches: torch.Tensor,
        quality: torch.Tensor,
        pad_mask: torch.Tensor,
        eye_nonmissing_frac: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits [B, out_dim] (1 for binary, num_classes for multiclass)."""
        hidden = self.bert.encode(
            stim_patches, eye_patches, pad_mask, eye_nonmissing_frac, task_ids
        )
        cls_out = hidden[:, 0, :]
        return self.classifier(cls_out)
