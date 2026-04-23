import pytest
from pathlib import Path

from dlecosys.shared.run import RunLayout


class TestRunLayoutPaths:
    def test_root_path(self, tmp_path):
        layout = RunLayout(str(tmp_path), "my_run")
        assert layout.root == tmp_path / "my_run"

    def test_expected_subdirs(self, tmp_path):
        layout = RunLayout(str(tmp_path), "run1")
        assert layout.data_dir == layout.root / "data"
        assert layout.checkpoints_dir == layout.root / "checkpoints"
        assert layout.logs_dir == layout.root / "logs"
        assert layout.predictions_dir == layout.root / "predictions"

    def test_expected_file_paths(self, tmp_path):
        layout = RunLayout(str(tmp_path), "run1")
        assert layout.config_path == layout.root / "config.yaml"
        assert layout.checkpoint_path == layout.checkpoints_dir / "best.pt"
        assert layout.history_path == layout.logs_dir / "history.csv"
        assert layout.summary_path == layout.logs_dir / "summary.json"
        assert layout.predictions_csv == layout.predictions_dir / "predictions.csv"

    def test_data_path_split(self, tmp_path):
        layout = RunLayout(str(tmp_path), "run1")
        assert layout.data_path("train") == layout.data_dir / "train.pt"
        assert layout.data_path("val") == layout.data_dir / "val.pt"
        assert layout.data_path("test") == layout.data_dir / "test.pt"


class TestRunLayoutCreate:
    def test_create_makes_all_directories(self, tmp_path):
        layout = RunLayout(str(tmp_path), "run1")
        layout.create()
        assert layout.data_dir.is_dir()
        assert layout.checkpoints_dir.is_dir()
        assert layout.logs_dir.is_dir()
        assert layout.predictions_dir.is_dir()

    def test_create_idempotent_when_no_config_written(self, tmp_path):
        layout = RunLayout(str(tmp_path), "run1")
        layout.create()
        layout.create()  # no config.yaml → should not raise

    def test_create_raises_if_config_exists_and_no_overwrite(self, tmp_path):
        layout = RunLayout(str(tmp_path), "run1")
        layout.create()
        layout.config_path.write_text("placeholder")
        with pytest.raises(FileExistsError, match="already exists"):
            layout.create(overwrite=False)

    def test_create_with_overwrite_succeeds(self, tmp_path):
        layout = RunLayout(str(tmp_path), "run1")
        layout.create()
        layout.config_path.write_text("placeholder")
        layout.create(overwrite=True)  # must not raise
        assert layout.data_dir.is_dir()
