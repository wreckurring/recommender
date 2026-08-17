"""Loss functions for implicit collaborative filtering and ranking optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) pairwise ranking loss.
    
    Optimizes maximum posterior probability of correctly ordering positive vs negative items:
    L_bpr = -ln(sigmoid(x_u,pos - x_u,neg)) + l2_reg * ||params||^2
    """

    def __init__(self, l2_reg: float = 1e-4) -> None:
        super().__init__()
        self.l2_reg = l2_reg

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        *regularized_params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise BPR loss with optional parameter regularization."""
        diff = pos_scores - neg_scores
        loss = -F.logsigmoid(diff).mean()

        if self.l2_reg > 0 and regularized_params:
            reg_term = torch.tensor(0.0, device=pos_scores.device)
            for param in regularized_params:
                reg_term += torch.sum(param**2)
            loss = loss + self.l2_reg * 0.5 * reg_term

        return loss


class PointwiseBCELoss(nn.Module):
    """Binary Cross Entropy with Logits for pointwise implicit interaction scoring."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)
