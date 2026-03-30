import io
import itertools
import logging
import random
import time

import numpy as np
import pandas as pd
import pytest
import pysdds
from pathlib import Path


cwd = Path(__file__).parent
root_sources = cwd / "files"


def to_str(slist):
    return [str(s) for s in slist]


ff = to_str((root_sources / "sources").glob("*"))


@pytest.mark.parametrize("file_root", ff)
def test_read_cols(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()
    for c in sdds.columns:
        for p in range(sdds.n_pages):
            col = sdds.col(c.name)
            a = col[p]
            b = col.data[p]
            assert np.array_equal(a, b)

    pairs = list(itertools.product(sdds.column_names, repeat=2))
    for random_cols in random.sample(pairs, min(len(pairs), 10)):
        sdds2 = pysdds.read(file_root, cols=random_cols)
        for c in sdds2.columns:
            if c._enabled:
                c2 = sdds.column_dict[c.name]
                c.compare(c2, raise_error=True)
            else:
                assert len(c.data) == 0


file_ts = [str(root_sources / "sources_compressed/timeSeries.sdds.xz")]
file_twi = [str(root_sources / "sources/twiss_binary")]


@pytest.mark.parametrize("file_root", file_ts)
def test_page_mask(file_root):
    sdds = pysdds.read(file_root)
    assert sdds.n_pages == 157
    sdds.columns_to_df(156)
    with pytest.raises(ValueError):
        sdds.columns_to_df(157)
    with pytest.raises(ValueError):
        sdds.columns_to_df(-1)

    sdds = pysdds.read(file_root, pages=[0])
    sdds.columns_to_df(0)
    assert sdds.n_pages == 1

    sdds = pysdds.read(file_root, pages=[0, 5])
    assert sdds.n_pages == 2
    sdds.columns_to_df(0)
    sdds.columns_to_df(1)
    with pytest.raises(ValueError):
        sdds.columns_to_df(2)


@pytest.mark.parametrize("file_root", file_ts)
def test_col_mask(file_root):
    sdds = pysdds.read(file_root, cols=["ReadbackName"])
    sdds.validate_data()
    assert sdds.n_pages == 157
    assert sdds.n_columns == 2
    assert len([c for c in sdds.columns if c._enabled]) == 1
    assert sdds.n_parameters == 11

    sdds = pysdds.read(file_root, cols=[1])
    assert sdds.n_pages == 157
    assert sdds.n_columns == 2
    assert not sdds.columns[0]._enabled
    assert sdds.columns[1]._enabled
    assert sdds.n_parameters == 11
    sdds.validate_data()


@pytest.mark.parametrize("file_root", file_ts)
def test_col_empty_shortcut(file_root):
    sdds = pysdds.read(file_root, cols=[], pages=[0])
    assert sdds.n_pages == 1
    assert sdds.n_columns == 2
    assert not sdds.columns[0]._enabled
    assert not sdds.columns[1]._enabled
    assert sdds.n_parameters == 11
    sdds.validate_data()

    sdds = pysdds.read(file_root, cols=[], pages=[1, 3])
    assert sdds.n_pages == 2
    assert sdds.n_columns == 2
    assert not sdds.columns[0]._enabled
    assert not sdds.columns[1]._enabled
    assert sdds.n_parameters == 11
    sdds.validate_data()


@pytest.mark.parametrize("file_root", file_twi)
def test_col_empty_perf(file_root):
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        # format='%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s', #[%(name)s]
        format="[%(levelname)-5.5s][%(asctime)s.%(msecs)03d %(filename)10s %(lineno)4s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    sdds = pysdds.read(file_root, cols=[], pages=[0])
    t1 = time.perf_counter()
    for i in range(10):
        sdds = pysdds.read(file_root, cols=[], pages=[0])
    t2 = time.perf_counter()
    for i in range(10):
        sdds = pysdds.read(file_root, pages=[0])
    t3 = time.perf_counter()
    logging.info(f"Empty cols read time: {t2 - t1:.6f}s vs full read time: {t3 - t2:.6f}s")
    assert t3 - t2 > t2 - t1
    assert sdds.n_pages == 1
    assert sdds.n_columns == 18
    sdds.validate_data()


def _make_multipage_sdds(mode, n_pages=5, mixed=False):
    """Create an n-page SDDS file in memory.

    mixed=False: all-numeric columns (routes to numeric parsers).
    mixed=True: includes a string column (routes to mixed/generic parsers).
    """
    if mixed:
        dfs = [
            pd.DataFrame({"x": np.arange(10, dtype=np.float64) + i * 100, "label": [f"p{i}r{j}" for j in range(10)]})
            for i in range(n_pages)
        ]
    else:
        dfs = [
            pd.DataFrame({"x": np.arange(10, dtype=np.float64) + i * 100, "y": np.arange(10, dtype=np.int32) + i * 200})
            for i in range(n_pages)
        ]
    params = {"idx": list(range(n_pages))}
    sdds = pysdds.SDDSFile.from_df(dfs, parameter_dict=params, mode=mode)
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    return buf, dfs


@pytest.mark.parametrize("mode", ["binary", "ascii"])
def test_page_select_non_first(mode):
    """Selecting a non-first page returns only that page."""
    buf, dfs = _make_multipage_sdds(mode)
    sdds = pysdds.read(io.BufferedReader(buf), pages=[2])
    assert sdds.n_pages == 1
    df = sdds.columns_to_df(0)
    assert np.array_equal(df["x"].values, dfs[2]["x"].values)
    assert np.array_equal(df["y"].values, dfs[2]["y"].values)


@pytest.mark.parametrize("mode", ["binary", "ascii"])
def test_page_select_sparse(mode):
    """Selecting non-contiguous pages returns only those pages."""
    buf, dfs = _make_multipage_sdds(mode)
    sdds = pysdds.read(io.BufferedReader(buf), pages=[1, 3])
    assert sdds.n_pages == 2
    for out_idx, src_idx in enumerate([1, 3]):
        df = sdds.columns_to_df(out_idx)
        assert np.array_equal(df["x"].values, dfs[src_idx]["x"].values)
        assert np.array_equal(df["y"].values, dfs[src_idx]["y"].values)


@pytest.mark.parametrize("mode", ["binary", "ascii"])
def test_page_select_last(mode):
    """Selecting only the last page returns one page."""
    buf, dfs = _make_multipage_sdds(mode)
    sdds = pysdds.read(io.BufferedReader(buf), pages=[4])
    assert sdds.n_pages == 1
    df = sdds.columns_to_df(0)
    assert np.array_equal(df["x"].values, dfs[4]["x"].values)


@pytest.mark.parametrize("mode", ["binary", "ascii"])
def test_page_select_mixed_columns(mode):
    """Page selection with string columns (mixed/generic parser path)."""
    buf, dfs = _make_multipage_sdds(mode, mixed=True)
    sdds = pysdds.read(io.BufferedReader(buf), pages=[2])
    assert sdds.n_pages == 1
    df = sdds.columns_to_df(0)
    assert np.array_equal(df["x"].values, dfs[2]["x"].values)
    assert np.array_equal(df["label"].values, dfs[2]["label"].values)


@pytest.mark.parametrize("mode", ["binary", "ascii"])
def test_page_select_no_columns(mode):
    """Page selection on a file with only parameters and no columns."""
    n_pages = 5
    dfs = [pd.DataFrame() for _ in range(n_pages)]
    params = {"idx": list(range(n_pages))}
    sdds = pysdds.SDDSFile.from_df(dfs, parameter_dict=params, mode=mode)
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf), pages=[2])
    assert sdds2.n_pages == 1
    assert sdds2.parameters[0].data[0] == 2
