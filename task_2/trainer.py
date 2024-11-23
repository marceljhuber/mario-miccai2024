# encoding: utf-8

"""
Trainer class to train a model for Task1 using two visits and
full OCT volumes.
"""

from utils.trainer_base import BaseTrainer
from utils.evaluator import Evaluator
from typing import Optional, Tuple
import torch
from torch.cuda.amp import GradScaler, autocast

class TwoVisitOCTTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()

    def forward(self,
                oct_t: torch.Tensor,
                oct_t1: torch.Tensor,
                fundus: Optional[torch.Tensor],
                y: torch.Tensor,
                df: Optional[dict],
                metric: callable,
                evaluator: Optional[Evaluator] = None,
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = self.device

        oct_t0 = oct_t.to(device)
        oct_t90 = oct_t1.to(device)

        with autocast():
            logits: torch.Tensor = self.model(oct_t0)
            loss: torch.Tensor = - metric(logits, oct_t90)

        y = torch.zeros(1,)

        return loss, (logits, y)