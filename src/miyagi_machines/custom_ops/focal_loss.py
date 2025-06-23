# src/miyagi_machines/custom_ops/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    focal loss (multi-class) with optional per-class or class-balanced weights

    args:

    alpha          scalar or Tensor[C].  If None → all ones.
    gamma          focusing parameter γ.
    reduction      "mean" | "sum" | "none"
    ignore_index   int or None.
    """
    def __init__(self,
                 alpha=None,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 ignore_index: int | None = None):
        super().__init__()
        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))  # broadcast later
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float))
        self.gamma        = gamma
        self.reduction    = reduction
        self.ignore_index = ignore_index

    @staticmethod
    def class_balanced_alpha(counts: torch.Tensor,
                             beta: float = 0.999) -> torch.Tensor:
        """
        Build Cui et al. (2019) class-balanced weights:
        α_c = (1-β) / (1-β^{n_c})
        """
        effective_num = 1.0 - torch.pow(beta, counts)
        return (1.0 - beta) / effective_num

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, C)
        targets: (B) int64
        """
        if logits.ndim != 2:
            raise ValueError("Expecting 2-D logits (B,C)")

        # Mask ignore_index
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            logits = logits[valid]
            targets = targets[valid]
            if targets.numel() == 0:          # all padded
                return logits.new_tensor(0.)

        logp  = F.log_softmax(logits, dim=-1)          # (B,C)
        p     = logp.exp()

        # Gather p_t and α_t
        pt    = p.gather(1, targets.unsqueeze(1)).squeeze(1)     # (B,)
        if self.alpha.ndim == 0:        # scalar
            at = self.alpha
        else:                           # tensor[C]
            at = self.alpha[targets]

        loss = -at * (1 - pt).pow(self.gamma) * logp.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
