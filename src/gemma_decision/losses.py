"""Losses on per-question option logits [Q, K] against gold (soft) distributions [Q, K].

soft CE   cross-entropy to the annotator distribution (KL + const), the main term
Brier     squared error of the probability vector; directly rewards calibration
ordinal   squared CDF distance (CRPS / EMD^2) on score questions: one level off costs less than three
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import TYPES


def decision_loss(
    logits, batch, brier: float = 1.0, ordinal: float = 0.5, hard: bool = False, hard_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """`hard_weight` blends the one-hot gold label into the soft target (0 = annotator distribution only,
    1 = label only; `hard=True` is the same as 1). More weight pushes argmax accuracy, less keeps calibration."""
    mask = batch["q_opt_mask"]
    target = batch["q_target"]
    w = 1.0 if hard else hard_weight
    if w > 0:
        target = (1 - w) * target + w * F.one_hot(batch["q_label"], target.shape[1]).float()

    logp = F.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    p = logp.exp() * mask
    ce = -(target * logp).sum(-1).mean()
    br = ((p - target) ** 2).sum(-1).mean()

    is_score = batch["q_type"] == TYPES["score"]
    if is_score.any():
        cdf = (p[is_score].cumsum(-1) - target[is_score].cumsum(-1)) ** 2
        ordl = (cdf * mask[is_score]).sum(-1).mean()
    else:
        ordl = logits.new_zeros(())

    total = ce + brier * br + ordinal * ordl
    return total, {"ce": ce.item(), "brier": br.item(), "ordinal": ordl.item()}
