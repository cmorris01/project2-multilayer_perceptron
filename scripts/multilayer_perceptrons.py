"""
Utility module for building Multilayer Perceptron (MLP) architectures used in the project.

The helpers below expose three ready-to-train networks:
    - baseline_mlp: 64 -> 32 -> 1
    - deep_feature_mlp: 128 -> 64 -> 32 -> 1
    - compact_batchnorm_mlp: 64 -> 32 -> 1 with BatchNorm on the hidden layers

The core builder is flexible enough to support additional architectures by supplying a custom
MLPConfig instance.
"""
import torch
from torch import nn


class BaselineMLP(nn.Module):
    """
    Baseline MLP architecture. Includes 2 hidden layers and 1 output layer; 64 nodes in the first hidden layer, use of 
    ReLU activation functions, dropout and L2 regularization.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size_1: int = 64, hidden_size_2: int = 32, dropout_rate: float = 0.4):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DeepFeatureMLP(nn.Module):
    """
    Deep feature MLP architecture. Includes 3 hidden layers and 1 output layer; 128 nodes in the first hidden layer, 64
    nodes in the second hidden layer, 32 nodes in the third hidden layer, and 1 output layer. Use of dropout and L2 regularization.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size_1: int = 128, hidden_size_2: int = 64, hidden_size_3: int = 32,
                 dropout_rate_1: float = 0.4, dropout_rate_2: float = 0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.layer4 = nn.Linear(hidden_size_3, output_size)
        self.dropout_1 = nn.Dropout(p=dropout_rate_1)
        self.dropout_2 = nn.Dropout(p=dropout_rate_2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.dropout_1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout_2(x)
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class CompactBatchnormMLP(nn.Module):
    """
    Compact MLP that leverages batch norm. Includes 2 hidden layers and 1 output layer; 64 nodes in the first hidden layer, use of
    ReLU activation functions, and batch normalization on the hidden layers.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size_1: int = 64, hidden_size_2: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x