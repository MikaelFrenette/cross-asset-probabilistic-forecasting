import pytest
import numpy as np
import torch
from dlecosys.shared.preprocessing import StandardScaler


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((100, 4))


@pytest.fixture
def fitted(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler, data


class TestStandardScalerFit:
    def test_zero_mean_after_transform(self, fitted):
        scaler, X = fitted
        np.testing.assert_allclose(scaler.transform(X).mean(axis=0), 0.0, atol=1e-10)

    def test_unit_std_after_transform(self, fitted):
        scaler, X = fitted
        np.testing.assert_allclose(scaler.transform(X).std(axis=0), 1.0, atol=1e-10)

    def test_inverse_roundtrip(self, fitted):
        scaler, X = fitted
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X, atol=1e-10)

    def test_n_features_in(self, fitted):
        scaler, _ = fitted
        assert scaler.n_features_in_ == 4


class TestStandardScalerErrors:
    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            StandardScaler().transform(np.ones((5, 4)))

    def test_wrong_feature_count_raises(self, fitted):
        scaler, _ = fitted
        with pytest.raises(ValueError, match="features"):
            scaler.transform(np.ones((5, 2)))

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2D"):
            StandardScaler().fit(np.ones(10))


class TestStandardScalerTypes:
    def test_tensor_input_returns_tensor(self, fitted):
        scaler, X = fitted
        out = scaler.transform(torch.from_numpy(X).float())
        assert isinstance(out, torch.Tensor)

    def test_fit_transform_matches_fit_then_transform(self, data):
        s1 = StandardScaler()
        s1.fit(data)
        expected = s1.transform(data)
        result = StandardScaler().fit_transform(data)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestStandardScalerPersistence:
    def test_save_load_roundtrip(self, fitted, tmp_path):
        scaler, X = fitted
        path = str(tmp_path / "scaler.pt")
        scaler.save(path)
        loaded = StandardScaler.load(path)
        np.testing.assert_allclose(scaler.transform(X), loaded.transform(X), atol=1e-12)

    def test_load_wrong_class_raises(self, tmp_path):
        import torch as _torch
        path = str(tmp_path / "bad.pt")
        _torch.save({"class": "SomethingElse", "state": {}}, path)
        with pytest.raises(TypeError):
            StandardScaler.load(path)

    def test_save_unfitted_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            StandardScaler().save(str(tmp_path / "scaler.pt"))
