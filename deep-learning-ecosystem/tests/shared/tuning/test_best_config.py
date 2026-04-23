"""
Verifies that the post-tuning best_config.yaml writing logic in scripts/tune.py
produces a valid, reloadable PipelineConfig with tuning removed and winning
params applied.
"""

from dlecosys.shared.config.schema import PipelineConfig
from dlecosys.shared.tuning.search_space import apply_suggestion, from_hashable


def _base_cfg_dict():
    return {
        "experiment": {"name": "r"},
        "data": {"task": "regression", "path": "d.csv"},
        "model": {"name": "mlp", "params": {"input_dim": 4}},
        "training": {
            "optimizer": {"name": "adam", "lr": 0.001},
            "loss": "mse",
        },
        "tuning": {
            "study_name": "s",
            "search_space": {"training.optimizer.lr": [1e-4, 1e-3, 1e-2]},
        },
    }


class TestBestConfigBuild:
    def test_applies_params_and_clears_tuning(self):
        cfg = PipelineConfig(**_base_cfg_dict())
        best_params = {"training.optimizer.lr": 0.01}

        best_cfg_dict = cfg.model_dump()
        best_cfg_dict["tuning"] = None
        for path, value in best_params.items():
            apply_suggestion(best_cfg_dict, path, from_hashable(value))
        best_cfg_dict["experiment"]["name"] = "s_best"

        # Must reload as a valid PipelineConfig
        reloaded = PipelineConfig(**best_cfg_dict)
        assert reloaded.tuning is None
        assert reloaded.training.optimizer.lr == 0.01
        assert reloaded.experiment.name == "s_best"

    def test_tuple_param_converts_to_list(self):
        """hidden_dims as list-of-list: Optuna stores tuple, best_config needs list."""
        cfg_dict = _base_cfg_dict()
        cfg_dict["model"]["params"]["hidden_dims"] = [32]
        cfg = PipelineConfig(**cfg_dict)

        best_params = {"model.params.hidden_dims": (64, 32)}  # Optuna-style tuple

        best_cfg_dict = cfg.model_dump()
        best_cfg_dict["tuning"] = None
        for path, value in best_params.items():
            apply_suggestion(best_cfg_dict, path, from_hashable(value))

        reloaded = PipelineConfig(**best_cfg_dict)
        assert reloaded.model.params["hidden_dims"] == [64, 32]
