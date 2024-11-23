# encoding: utf-8

"""
File for PyTorch models for Task1.

Currently only contains a tiny model for projecting the
feature extractions to the classes.
"""

from torch import nn
import torch.nn.init as init
import torch


class LinearProjection(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int = 4,
                 bias: bool = True,
                 dropout_p: float = 0.5
                 ) -> None:
        super(LinearProjection, self).__init__()

        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear2 = nn.Linear(hidden_features, hidden_features, bias=bias)
        self.linear3 = nn.Linear(hidden_features, out_features, bias=bias)

        self.dropout = nn.Dropout(dropout_p)

        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.dropout(self.linear1(x)))
        x = self.activation(self.dropout(self.linear2(x)))
        return self.linear3(x)
