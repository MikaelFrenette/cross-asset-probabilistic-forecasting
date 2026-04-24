import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return nn.Linear(4, 1)


@pytest.fixture
def tiny_dataset():
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    y = torch.randn(20, 1)
    return TensorDataset(X, y)


@pytest.fixture
def tiny_loader(tiny_dataset):
    return DataLoader(tiny_dataset, batch_size=5)


@pytest.fixture
def tiny_val_loader(tiny_dataset):
    return DataLoader(tiny_dataset, batch_size=5)
