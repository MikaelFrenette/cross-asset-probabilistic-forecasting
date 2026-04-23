"""
Run Layout
----------
Defines and enforces the non-negotiable output directory structure for every
pipeline run.

Classes
-------
DataPaths
    Protocol describing the data-side API shared by RunLayout, StudyLayout,
    and EnsembleLayout. Used to type preprocess.py's layout argument generically.
RunLayout
    Encapsulates all output paths for a single pipeline run.
StudyLayout
    Output layout for a hyperparameter tuning study. Holds study-level
    artifacts and constructs per-trial RunLayouts that share the study
    data directory.
EnsembleLayout
    Output layout for a bagging ensemble run. Shares a raw preprocessed pool
    across estimators; each estimator gets its own RunLayout with its own
    checkpoint, logs, and persisted scaler/tokenizers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

__all__ = ["DataPaths", "RunLayout", "StudyLayout", "EnsembleLayout"]


@runtime_checkable
class DataPaths(Protocol):
    """Shared data-side contract between RunLayout and StudyLayout.

    Any layout that writes preprocessed splits must expose these paths.
    preprocess.py accepts anything that satisfies this protocol, which
    makes RunLayout and StudyLayout drop-in replacements for one another
    at the data-writing boundary.
    """

    data_dir: Path
    tokenizers_dir: Path
    scaler_path: Path
    config_path: Path

    def data_path(self, split: str) -> Path: ...
    def tokenizer_path(self, col_idx: int) -> Path: ...
    def create(self, *, overwrite: bool = False) -> None: ...


class RunLayout:
    """
    Output directory contract for a single pipeline run.

    Every run produces the same subdirectory structure under
    ``<output_dir>/<run_id>/``. This class is the single source of truth for
    all output paths — scripts never construct paths manually.

    Parameters
    ----------
    output_dir : str
        Root output directory (e.g. ``"outputs"``).
    run_id : str
        Unique run identifier set in the experiment config.
    data_dir : Path, optional
        External data directory to use instead of ``<root>/data``. Used by
        tuning trials, which share preprocessed data at the study level
        rather than keeping a per-trial copy.

    Directory layout
    ----------------
    ::

        <output_dir>/
            <run_id>/
                config.yaml
                data/
                    train.pt
                    val.pt
                    test.pt
                checkpoints/
                    best.pt
                logs/
                    history.csv
                    summary.json
                predictions/
                    predictions.csv
    """

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        *,
        data_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.root = Path(output_dir) / run_id
        self.data_dir = Path(data_dir) if data_dir is not None else self.root / "data"
        self.checkpoints_dir = self.root / "checkpoints"
        self.logs_dir = self.root / "logs"
        self.predictions_dir = self.root / "predictions"

        self.config_path = self.root / "config.yaml"
        self.checkpoint_path = self.checkpoints_dir / "best.pt"
        self.scaler_path = self.data_dir / "scaler.pt"
        self.tokenizers_dir = self.data_dir / "tokenizers"
        self.history_path = self.logs_dir / "history.csv"
        self.summary_path = self.logs_dir / "summary.json"
        self.predictions_csv = self.predictions_dir / "predictions.csv"

    def data_path(self, split: str) -> Path:
        """Return the path for a data split file (train / val / test)."""
        return self.data_dir / f"{split}.pt"

    def tokenizer_path(self, col_idx: int) -> Path:
        """Return the path for a fitted tokenizer for a given column index."""
        return self.tokenizers_dir / f"tokenizer_col{col_idx}.pt"

    def fold_layout(self, fold_idx: int) -> "RunLayout":
        """Return a RunLayout for a specific fold, sharing this layout's data dir.

        Used by tuning when ``splitter.n_splits > 1`` so each fold's
        checkpoints and logs land under ``<self.root>/fold_{fold_idx}/``
        instead of overwriting each other.
        """
        return RunLayout(
            output_dir=str(self.root),
            run_id=f"fold_{fold_idx}",
            data_dir=self.data_dir,
        )

    def create(self, *, overwrite: bool = False) -> None:
        """
        Create the full directory tree for this run.

        Parameters
        ----------
        overwrite : bool, default False
            If False and the run directory already contains a ``config.yaml``
            (i.e. the run was previously initialized), raise ``FileExistsError``.

        Raises
        ------
        FileExistsError
            If the run already exists and ``overwrite`` is False.
        """
        if self.config_path.exists() and not overwrite:
            raise FileExistsError(
                f"Run '{self.root}' already exists. "
                f"Pass --overwrite to overwrite."
            )
        for directory in (
            self.data_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.predictions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class StudyLayout:
    """
    Output directory contract for a hyperparameter tuning study.

    A study groups N trials that share a preprocessed dataset. Study-level
    artifacts (best params, trials index, optional sqlite storage) live at
    the study root; each trial has its own RunLayout underneath with its
    own checkpoints and logs, but shares the study's data directory.

    Directory layout
    ----------------
    ::

        <output_dir>/
            <study_name>/
                config.yaml
                best_params.yaml
                trials.csv
                study.db              # only when storage is sqlite
                data/                 # shared across all trials
                    train.pt / val.pt / test.pt / scaler.pt / tokenizers/
                trial_0/
                    config.yaml
                    checkpoints/best.pt
                    logs/history.csv, summary.json
                trial_1/
                    ...
    """

    def __init__(self, output_dir: str, study_name: str) -> None:
        self.root = Path(output_dir) / study_name
        self.data_dir = self.root / "data"
        self.tokenizers_dir = self.data_dir / "tokenizers"
        self.scaler_path = self.data_dir / "scaler.pt"

        self.config_path = self.root / "config.yaml"
        self.best_params_path = self.root / "best_params.yaml"
        self.best_config_path = self.root / "best_config.yaml"
        self.trials_csv = self.root / "trials.csv"
        self.storage_path = self.root / "study.db"

    def data_path(self, split: str) -> Path:
        return self.data_dir / f"{split}.pt"

    def tokenizer_path(self, col_idx: int) -> Path:
        return self.tokenizers_dir / f"tokenizer_col{col_idx}.pt"

    def trial_layout(self, trial_number: int) -> RunLayout:
        """Construct a RunLayout for a trial that shares this study's data dir."""
        return RunLayout(
            output_dir=str(self.root),
            run_id=f"trial_{trial_number}",
            data_dir=self.data_dir,
        )

    def create(self, *, overwrite: bool = False) -> None:
        if self.config_path.exists() and not overwrite:
            raise FileExistsError(
                f"Study '{self.root}' already exists. Pass --overwrite to overwrite."
            )
        self.data_dir.mkdir(parents=True, exist_ok=True)


class EnsembleLayout:
    """
    Output directory contract for a bagging ensemble run.

    Shares a raw preprocessed pool (``data/``) across every estimator.
    Each estimator has its own RunLayout under ``ensemble/estimator_{id}/``
    with its own checkpoint, logs, and persisted scaler/tokenizers. Final
    aggregated test predictions land in ``predictions/predictions.csv``.

    Directory layout
    ----------------
    ::

        <output_dir>/
            <run_id>/
                config.yaml
                data/                          # shared raw pool + test
                    train.pt / val.pt / test.pt
                ensemble/
                    estimator_0/
                        config.yaml
                        checkpoints/best.pt
                        logs/history.csv, summary.json
                        scaler.pt              # per-estimator, fit on bootstrap
                        tokenizers/...         # per-estimator
                        sample_indices.pt      # bootstrap sample indices
                    estimator_1/...
                predictions/
                    predictions.csv            # aggregated across estimators
    """

    def __init__(self, output_dir: str, run_id: str) -> None:
        self.root = Path(output_dir) / run_id
        self.data_dir = self.root / "data"
        self.tokenizers_dir = self.data_dir / "tokenizers"
        self.scaler_path = self.data_dir / "scaler.pt"
        self.config_path = self.root / "config.yaml"
        self.ensemble_dir = self.root / "ensemble"
        self.selected_path = self.ensemble_dir / "selected.json"
        self.predictions_dir = self.root / "predictions"
        self.predictions_csv = self.predictions_dir / "predictions.csv"

    def data_path(self, split: str) -> Path:
        return self.data_dir / f"{split}.pt"

    def tokenizer_path(self, col_idx: int) -> Path:
        return self.tokenizers_dir / f"tokenizer_col{col_idx}.pt"

    def estimator_layout(self, estimator_id: int) -> RunLayout:
        """Return a RunLayout rooted at ensemble/estimator_{id}/, sharing this
        layout's data dir so bootstrapping reads the same raw pool."""
        return RunLayout(
            output_dir=str(self.ensemble_dir),
            run_id=f"estimator_{estimator_id}",
            data_dir=self.data_dir,
        )

    def estimator_scaler_path(self, estimator_id: int) -> Path:
        """Per-estimator scaler, co-located with the checkpoint (NOT in shared data_dir)."""
        return self.ensemble_dir / f"estimator_{estimator_id}" / "scaler.pt"

    def estimator_tokenizers_dir(self, estimator_id: int) -> Path:
        return self.ensemble_dir / f"estimator_{estimator_id}" / "tokenizers"

    def estimator_sample_indices_path(self, estimator_id: int) -> Path:
        return self.ensemble_dir / f"estimator_{estimator_id}" / "sample_indices.pt"

    def estimator_feature_indices_path(self, estimator_id: int) -> Path:
        return self.ensemble_dir / f"estimator_{estimator_id}" / "feature_indices.pt"

    def create(self, *, overwrite: bool = False) -> None:
        if self.config_path.exists() and not overwrite:
            raise FileExistsError(
                f"Ensemble '{self.root}' already exists. Pass --overwrite to overwrite."
            )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
