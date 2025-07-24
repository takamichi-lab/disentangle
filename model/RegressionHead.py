import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from typing import List, Optional

class RegressionHead_for_physicalValue(nn.Module):
    """
    部屋の面積,
    距離,
    方向,
    残響時間を予測するヘッド
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    