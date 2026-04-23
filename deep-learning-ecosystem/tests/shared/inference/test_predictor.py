import pytest
import numpy as np
import torch
import torch.nn as nn

from dlecosys.shared.inference import Predictor


def test_predict_from_numpy():
    model = nn.Linear(4, 1)
    predictor = Predictor(model)
    X = np.random.randn(10, 4).astype(np.float32)
    out = predictor.predict(X)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (10, 1)


def test_predict_from_tensor():
    model = nn.Linear(4, 2)
    predictor = Predictor(model)
    out = predictor.predict(torch.randn(8, 4))
    assert out.shape == (8, 2)


def test_predict_output_on_cpu():
    model = nn.Linear(4, 1)
    predictor = Predictor(model)
    out = predictor.predict(torch.randn(5, 4))
    assert out.device.type == "cpu"


def test_predict_batching_is_consistent():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    predictor = Predictor(model)
    X = torch.randn(7, 3)
    out_small = predictor.predict(X, batch_size=2)
    out_large = predictor.predict(X, batch_size=100)
    assert torch.allclose(out_small, out_large, atol=1e-5)


def test_from_checkpoint_loads_weights(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    path = str(tmp_path / "ckpt.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 0, "best_metric": 0.5},
        path,
    )
    new_model = nn.Linear(4, 1)
    predictor = Predictor.from_checkpoint(path, new_model)
    assert torch.allclose(predictor.model.weight.data, model.weight.data)


def test_predict_with_preprocessor():
    from dlecosys.shared.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((50, 3)).astype(np.float64)
    scaler = StandardScaler()
    scaler.fit(X_train)

    model = nn.Linear(3, 1)
    predictor = Predictor(model, preprocessors=[scaler])

    X_test = rng.standard_normal((10, 3))
    out = predictor.predict(X_test)
    assert out.shape == (10, 1)
