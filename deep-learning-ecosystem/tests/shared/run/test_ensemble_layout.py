import pytest

from dlecosys.shared.run import DataPaths, EnsembleLayout, RunLayout


class TestEnsembleLayout:
    def test_paths(self, tmp_path):
        e = EnsembleLayout(str(tmp_path), "myrun")
        assert e.root == tmp_path / "myrun"
        assert e.data_dir == tmp_path / "myrun" / "data"
        assert e.ensemble_dir == tmp_path / "myrun" / "ensemble"
        assert e.predictions_dir == tmp_path / "myrun" / "predictions"
        assert e.predictions_csv == tmp_path / "myrun" / "predictions" / "predictions.csv"
        assert e.config_path == tmp_path / "myrun" / "config.yaml"
        assert e.selected_path == tmp_path / "myrun" / "ensemble" / "selected.json"

    def test_data_path(self, tmp_path):
        e = EnsembleLayout(str(tmp_path), "myrun")
        assert e.data_path("train") == tmp_path / "myrun" / "data" / "train.pt"

    def test_estimator_layout_shares_data_dir(self, tmp_path):
        e = EnsembleLayout(str(tmp_path), "myrun")
        est = e.estimator_layout(3)
        assert isinstance(est, RunLayout)
        assert est.root == tmp_path / "myrun" / "ensemble" / "estimator_3"
        assert est.data_dir == e.data_dir

    def test_per_estimator_paths_are_unique(self, tmp_path):
        e = EnsembleLayout(str(tmp_path), "myrun")
        p0 = e.estimator_scaler_path(0)
        p1 = e.estimator_scaler_path(1)
        assert p0 != p1
        assert p0 == tmp_path / "myrun" / "ensemble" / "estimator_0" / "scaler.pt"

    def test_satisfies_data_paths_protocol(self, tmp_path):
        e = EnsembleLayout(str(tmp_path), "myrun")
        assert isinstance(e, DataPaths)

    def test_create(self, tmp_path):
        e = EnsembleLayout(str(tmp_path), "myrun")
        e.create()
        assert e.data_dir.exists()
        assert e.ensemble_dir.exists()
        assert e.predictions_dir.exists()

    def test_create_raises_if_exists(self, tmp_path):
        e = EnsembleLayout(str(tmp_path), "myrun")
        e.create()
        e.config_path.write_text("existing")
        with pytest.raises(FileExistsError):
            e.create(overwrite=False)
