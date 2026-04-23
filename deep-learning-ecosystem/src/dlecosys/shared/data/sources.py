"""
Tabular Data Sources
--------------------
File-format dispatcher for the preprocessing lane. Reads the file at
``cfg.data.path``, infers the loader from the extension, and returns a
``pandas.DataFrame``.

Bundled loaders
---------------
.csv / .tsv / .txt    → pandas.read_csv  (separator inferred from extension)
.parquet / .pq        → pandas.read_parquet (requires pyarrow or fastparquet)

Extending
---------
To add a new source (say JSON-Lines), append a branch::

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

Then preprocess.py picks it up automatically via the extension.

For sources that don't map to a single file (SQL, object store, streaming),
the cleanest path is to pre-materialize into a local Parquet file and point
``cfg.data.path`` at that file. Keeping a single "path on disk" contract in
the pipeline avoids leaking credentials / connection strings into
``config.yaml``.

Functions
---------
load_tabular
    Read the file at ``path`` into a DataFrame, dispatching by extension.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_tabular"]

_CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


def load_tabular(path: str) -> pd.DataFrame:
    """Read a tabular file into a DataFrame, dispatching on file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    suffix = p.suffix.lower()

    if suffix in _CSV_SUFFIXES:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep)

    if suffix in _PARQUET_SUFFIXES:
        try:
            return pd.read_parquet(p)
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires 'pyarrow' or 'fastparquet'. "
                "Install one with: pip install pyarrow"
            ) from exc

    raise ValueError(
        f"Unsupported data file extension: {suffix!r}. "
        f"Supported: {sorted(_CSV_SUFFIXES | _PARQUET_SUFFIXES)}. "
        f"See src/dlecosys/shared/data/sources.py to add a new format."
    )
