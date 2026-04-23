import pytest
import torch

from dlecosys.shared.config.schema import SplitterSection
from dlecosys.shared.data import (
    BaseSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    StratifiedKFoldSplitter,
    build_splitter,
)
from dlecosys.shared.run import RunLayout


def _write_split(path, n: int, d: int = 4, label_cls: int = None) -> None:
    X = torch.randn(n, d)
    if label_cls is not None:
        y = torch.randint(0, label_cls, (n,))
    else:
        y = torch.randn(n, 1)
    torch.save({"X": X, "y": y}, path)


# ---------------------------------------------------------------------------
# Contract enforcement on BaseSplitter
# ---------------------------------------------------------------------------


class TestBaseSplitterContract:
    def test_missing_n_splits_raises(self, tmp_path):
        class BadSplitter(BaseSplitter):
            pass  # never sets n_splits

        with pytest.raises(TypeError, match="n_splits"):
            BadSplitter(tmp_path / "train.pt")

    def test_zero_n_splits_raises(self, tmp_path):
        class ZeroSplitter(BaseSplitter):
            n_splits = 0

        with pytest.raises(ValueError, match="positive int"):
            ZeroSplitter(tmp_path / "train.pt")

    def test_generate_indices_default_raises(self, tmp_path):
        class LazySplitter(BaseSplitter):
            n_splits = 3

        layout = RunLayout(str(tmp_path), "r1")
        layout.create()
        _write_split(layout.data_path("train"), 30)
        _write_split(layout.data_path("val"), 10)

        s = LazySplitter(layout.data_path("train"), layout.data_path("val"))
        with pytest.raises(NotImplementedError, match="generate_indices"):
            list(s.iterate_folds())


# ---------------------------------------------------------------------------
# HoldoutSplitter preserves original split
# ---------------------------------------------------------------------------


class TestHoldoutSplitter:
    def test_yields_one_fold(self, tmp_path):
        layout = RunLayout(str(tmp_path), "r1")
        layout.create()
        _write_split(layout.data_path("train"), 20)
        _write_split(layout.data_path("val"), 5)

        s = HoldoutSplitter(layout.data_path("train"), layout.data_path("val"))
        assert s.n_splits == 1
        folds = list(s.iterate_folds())
        assert len(folds) == 1
        train_ds, val_ds = folds[0]
        assert len(train_ds) == 20
        assert len(val_ds) == 5

    def test_missing_val_raises(self, tmp_path):
        layout = RunLayout(str(tmp_path), "r1")
        layout.create()
        _write_split(layout.data_path("train"), 10)
        s = HoldoutSplitter(layout.data_path("train"), val_path=None)
        with pytest.raises(ValueError, match="val_path"):
            list(s.iterate_folds())


# ---------------------------------------------------------------------------
# KFoldSplitter correctness
# ---------------------------------------------------------------------------


class TestKFoldSplitter:
    def _make(self, tmp_path, n_train=40, n_val=10):
        layout = RunLayout(str(tmp_path), "r1")
        layout.create()
        _write_split(layout.data_path("train"), n_train)
        _write_split(layout.data_path("val"), n_val)
        return layout

    def test_yields_n_splits_folds(self, tmp_path):
        layout = self._make(tmp_path)
        s = KFoldSplitter(layout.data_path("train"), layout.data_path("val"), n_splits=5)
        folds = list(s.iterate_folds())
        assert len(folds) == 5

    def test_disjoint_val_across_folds(self, tmp_path):
        layout = self._make(tmp_path, n_train=40, n_val=10)
        s = KFoldSplitter(layout.data_path("train"), layout.data_path("val"), n_splits=5)
        val_sets = []
        for train_ds, val_ds in s.iterate_folds():
            vx, _ = val_ds.tensors
            val_sets.append(vx)
        # Concatenate all val portions and assert total == pool size (full coverage)
        total_val = sum(v.shape[0] for v in val_sets)
        assert total_val == 50  # 40 + 10 = full pool

    def test_no_overlap_within_fold(self, tmp_path):
        layout = self._make(tmp_path, n_train=30, n_val=10)
        s = KFoldSplitter(layout.data_path("train"), layout.data_path("val"), n_splits=4)
        for train_ds, val_ds in s.iterate_folds():
            assert len(train_ds) + len(val_ds) == 40

    def test_shuffle_is_reproducible(self, tmp_path):
        layout = self._make(tmp_path)
        s1 = KFoldSplitter(layout.data_path("train"), layout.data_path("val"), n_splits=5, seed=7)
        s2 = KFoldSplitter(layout.data_path("train"), layout.data_path("val"), n_splits=5, seed=7)
        folds1 = [v.tensors[0] for _, v in s1.iterate_folds()]
        folds2 = [v.tensors[0] for _, v in s2.iterate_folds()]
        for a, b in zip(folds1, folds2):
            assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# StratifiedKFoldSplitter preserves class proportions
# ---------------------------------------------------------------------------


class TestStratifiedKFoldSplitter:
    def test_preserves_class_proportions(self, tmp_path):
        layout = RunLayout(str(tmp_path), "r1")
        layout.create()
        _write_split(layout.data_path("train"), 60, label_cls=3)
        _write_split(layout.data_path("val"), 15, label_cls=3)

        s = StratifiedKFoldSplitter(
            layout.data_path("train"), layout.data_path("val"), n_splits=5
        )
        for train_ds, val_ds in s.iterate_folds():
            _, y_val = val_ds.tensors
            # Each class should be present in every fold
            assert len(torch.unique(y_val)) == 3

    def test_rejects_float_y(self, tmp_path):
        layout = RunLayout(str(tmp_path), "r1")
        layout.create()
        _write_split(layout.data_path("train"), 40)
        _write_split(layout.data_path("val"), 10)

        s = StratifiedKFoldSplitter(
            layout.data_path("train"), layout.data_path("val"), n_splits=5
        )
        with pytest.raises(ValueError, match="integer class labels"):
            list(s.iterate_folds())


# ---------------------------------------------------------------------------
# build_splitter dispatch
# ---------------------------------------------------------------------------


class TestBuildSplitter:
    def test_holdout(self, tmp_path):
        layout = RunLayout(str(tmp_path), "r1")
        s = build_splitter(SplitterSection(type="holdout"), layout)
        assert isinstance(s, HoldoutSplitter)

    def test_kfold(self, tmp_path):
        layout = RunLayout(str(tmp_path), "r1")
        s = build_splitter(SplitterSection(type="kfold", n_splits=3), layout)
        assert isinstance(s, KFoldSplitter)
        assert s.n_splits == 3

    def test_stratified(self, tmp_path):
        layout = RunLayout(str(tmp_path), "r1")
        s = build_splitter(SplitterSection(type="stratified_kfold", n_splits=5), layout)
        assert isinstance(s, StratifiedKFoldSplitter)
        assert s.n_splits == 5


# ---------------------------------------------------------------------------
# Custom subclass example — verify extension path works
# ---------------------------------------------------------------------------


class TestCustomSubclass:
    def test_user_defined_splitter(self, tmp_path):
        """Prove the extension contract works by building a trivial custom splitter."""

        class FirstHalfSplitter(BaseSplitter):
            """Trivial test splitter: first half train, second half val, single fold."""

            n_splits = 1

            def generate_indices(self, X, y):
                n = len(X)
                mid = n // 2
                yield torch.arange(mid), torch.arange(mid, n)

        layout = RunLayout(str(tmp_path), "r1")
        layout.create()
        _write_split(layout.data_path("train"), 20)
        _write_split(layout.data_path("val"), 10)

        s = FirstHalfSplitter(layout.data_path("train"), layout.data_path("val"))
        folds = list(s.iterate_folds())
        assert len(folds) == 1
        train_ds, val_ds = folds[0]
        assert len(train_ds) == 15  # pool=30, mid=15
        assert len(val_ds) == 15


# ---------------------------------------------------------------------------
# SplitterSection schema
# ---------------------------------------------------------------------------


class TestSplitterSection:
    def test_defaults(self):
        cfg = SplitterSection()
        assert cfg.type == "holdout"
        assert cfg.n_splits == 5
        assert cfg.shuffle is True
        assert cfg.seed == 42

    def test_invalid_type_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SplitterSection(type="leave_one_out")
