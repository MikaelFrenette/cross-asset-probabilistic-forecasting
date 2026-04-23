import pytest
import numpy as np
from dlecosys.shared.preprocessing import IntegerTokenizer, StringTokenizer


class TestIntegerTokenizer:
    def test_fit_transform_roundtrip(self):
        tok = IntegerTokenizer()
        X = np.array([10, 20, 30, 10])
        ids = tok.fit_transform(X)
        back = tok.inverse_transform(ids)
        np.testing.assert_array_equal(back, X)

    def test_vocab_sorted_numerically(self):
        tok = IntegerTokenizer()
        tok.fit(np.array([30, 10, 20]))
        assert tok._vocab[10] < tok._vocab[20] < tok._vocab[30]

    def test_unknown_error_mode_raises(self):
        tok = IntegerTokenizer(handle_unknown="error")
        tok.fit(np.array([1, 2, 3]))
        with pytest.raises(KeyError):
            tok.transform(np.array([99]))

    def test_unknown_unk_mode_maps_to_zero(self):
        tok = IntegerTokenizer(handle_unknown="unk")
        tok.fit(np.array([1, 2, 3]))
        assert tok.transform(np.array([99]))[0] == 0

    def test_vocab_size_includes_unk(self):
        tok = IntegerTokenizer(handle_unknown="unk")
        tok.fit(np.array([1, 2, 3]))
        assert tok.vocab_size == 4  # 3 values + <unk>

    def test_vocab_size_no_unk(self):
        tok = IntegerTokenizer(handle_unknown="error")
        tok.fit(np.array([1, 2, 3]))
        assert tok.vocab_size == 3

    def test_non_integer_dtype_raises(self):
        with pytest.raises(TypeError):
            IntegerTokenizer().fit(np.array([1.5, 2.5]))

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            IntegerTokenizer().transform(np.array([1]))

    def test_save_load_roundtrip(self, tmp_path):
        tok = IntegerTokenizer(handle_unknown="unk")
        tok.fit(np.array([10, 20, 30]))
        path = str(tmp_path / "tok.pt")
        tok.save(path)
        loaded = IntegerTokenizer.load(path)
        np.testing.assert_array_equal(
            tok.transform(np.array([10, 99])),
            loaded.transform(np.array([10, 99])),
        )

    def test_invalid_handle_unknown_raises(self):
        with pytest.raises(ValueError):
            IntegerTokenizer(handle_unknown="ignore")


class TestStringTokenizer:
    def test_fit_transform_roundtrip(self):
        tok = StringTokenizer()
        X = np.array(["cat", "dog", "cat"])
        ids = tok.fit_transform(X)
        back = tok.inverse_transform(ids)
        np.testing.assert_array_equal(back, X)

    def test_vocab_sorted_alphabetically(self):
        tok = StringTokenizer()
        tok.fit(np.array(["zebra", "apple", "mango"]))
        assert tok._vocab["apple"] < tok._vocab["mango"] < tok._vocab["zebra"]

    def test_unknown_unk_mode(self):
        tok = StringTokenizer(handle_unknown="unk")
        tok.fit(np.array(["cat", "dog"]))
        assert tok.transform(np.array(["bird"]))[0] == 0

    def test_non_string_dtype_raises(self):
        with pytest.raises(TypeError):
            StringTokenizer().fit(np.array([1, 2, 3]))

    def test_save_load_roundtrip(self, tmp_path):
        tok = StringTokenizer()
        tok.fit(np.array(["a", "b", "c"]))
        path = str(tmp_path / "tok.pt")
        tok.save(path)
        loaded = StringTokenizer.load(path)
        np.testing.assert_array_equal(
            tok.transform(np.array(["a", "c"])),
            loaded.transform(np.array(["a", "c"])),
        )
