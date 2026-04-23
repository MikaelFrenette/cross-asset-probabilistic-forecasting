import pytest
import pandas as pd

from dlecosys.shared.data import load_tabular


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "target": [10, 20, 30]})


class TestLoadTabular:
    def test_csv(self, tmp_path, sample_df):
        p = tmp_path / "data.csv"
        sample_df.to_csv(p, index=False)
        df = load_tabular(str(p))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_df)

    def test_tsv_uses_tab_separator(self, tmp_path, sample_df):
        p = tmp_path / "data.tsv"
        sample_df.to_csv(p, index=False, sep="\t")
        df = load_tabular(str(p))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_df)

    def test_txt_treated_as_csv(self, tmp_path, sample_df):
        p = tmp_path / "data.txt"
        sample_df.to_csv(p, index=False)
        df = load_tabular(str(p))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_df)

    def test_parquet(self, tmp_path, sample_df):
        pytest.importorskip("pyarrow")
        p = tmp_path / "data.parquet"
        sample_df.to_parquet(p, index=False)
        df = load_tabular(str(p))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_df)

    def test_pq_extension(self, tmp_path, sample_df):
        pytest.importorskip("pyarrow")
        p = tmp_path / "data.pq"
        sample_df.to_parquet(p, index=False)
        df = load_tabular(str(p))
        pd.testing.assert_frame_equal(df.reset_index(drop=True), sample_df)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tabular(str(tmp_path / "nothing.csv"))

    def test_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported data file extension"):
            load_tabular(str(p))
