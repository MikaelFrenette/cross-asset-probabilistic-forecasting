import numpy as np
import torch
from dlecosys.shared.reproducibility import seed_everything


def test_same_seed_produces_same_torch_output():
    seed_everything(42)
    x1 = torch.randn(10)
    seed_everything(42)
    x2 = torch.randn(10)
    assert torch.allclose(x1, x2)


def test_same_seed_produces_same_numpy_output():
    seed_everything(7)
    a1 = np.random.randn(10)
    seed_everything(7)
    a2 = np.random.randn(10)
    np.testing.assert_array_equal(a1, a2)


def test_different_seeds_produce_different_output():
    seed_everything(1)
    x1 = torch.randn(10)
    seed_everything(2)
    x2 = torch.randn(10)
    assert not torch.allclose(x1, x2)
