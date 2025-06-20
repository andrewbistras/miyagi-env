# /src/miyagi_machines/custom_ops/sup_con_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Supervised contrastive loss ---------- #

class SupConLoss(nn.Module):
    """Implementation is vaguely from https://arxiv.org/abs/2004.11362 """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, reps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize once
        reps = F.normalize(reps, dim=1)
        sim = torch.mm(reps, reps.t()) / self.t

        # Create mask 
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        # Numerical stability with detach
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()
        
        # Efficient exp and log computation
        exp_sim = torch.exp(sim) * (1 - torch.eye(len(reps), device=reps.device))
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute loss only on positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        
        return loss
