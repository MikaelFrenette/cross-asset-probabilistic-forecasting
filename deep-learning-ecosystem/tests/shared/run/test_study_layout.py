import pytest

from dlecosys.shared.run import DataPaths, RunLayout, StudyLayout


class TestStudyLayout:
    def test_root_and_paths(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        assert s.root == tmp_path / "my_study"
        assert s.data_dir == tmp_path / "my_study" / "data"
        assert s.config_path == tmp_path / "my_study" / "config.yaml"
        assert s.best_params_path == tmp_path / "my_study" / "best_params.yaml"
        assert s.best_config_path == tmp_path / "my_study" / "best_config.yaml"
        assert s.trials_csv == tmp_path / "my_study" / "trials.csv"
        assert s.storage_path == tmp_path / "my_study" / "study.db"

    def test_data_path(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        assert s.data_path("train") == tmp_path / "my_study" / "data" / "train.pt"
        assert s.data_path("val") == tmp_path / "my_study" / "data" / "val.pt"

    def test_tokenizer_path(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        assert (
            s.tokenizer_path(3)
            == tmp_path / "my_study" / "data" / "tokenizers" / "tokenizer_col3.pt"
        )

    def test_create_idempotent(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        s.create()
        assert s.data_dir.exists()

    def test_create_raises_if_exists(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        s.create()
        s.config_path.write_text("existing")
        with pytest.raises(FileExistsError):
            s.create(overwrite=False)

    def test_create_overwrite_ok(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        s.create()
        s.config_path.write_text("existing")
        s.create(overwrite=True)  # must not raise


class TestDataPathsProtocol:
    def test_run_layout_satisfies(self, tmp_path):
        r = RunLayout(str(tmp_path), "r1")
        assert isinstance(r, DataPaths)

    def test_study_layout_satisfies(self, tmp_path):
        s = StudyLayout(str(tmp_path), "s1")
        assert isinstance(s, DataPaths)


class TestTrialLayoutSharesData:
    def test_trial_layout_points_to_study_data_dir(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        trial = s.trial_layout(5)
        assert isinstance(trial, RunLayout)
        assert trial.root == tmp_path / "my_study" / "trial_5"
        assert trial.data_dir == s.data_dir  # shared!
        assert trial.checkpoints_dir == tmp_path / "my_study" / "trial_5" / "checkpoints"
        assert trial.logs_dir == tmp_path / "my_study" / "trial_5" / "logs"

    def test_trial_data_path_uses_shared_dir(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        trial = s.trial_layout(0)
        assert trial.data_path("train") == s.data_path("train")

    def test_trial_checkpoint_is_per_trial(self, tmp_path):
        s = StudyLayout(str(tmp_path), "my_study")
        t0 = s.trial_layout(0)
        t1 = s.trial_layout(1)
        assert t0.checkpoint_path != t1.checkpoint_path
