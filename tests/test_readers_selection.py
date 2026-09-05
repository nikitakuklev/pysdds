"""Differential check of column/page/array selection against a full read.

Every cell value encodes its (page, column, row), so any swapped or shifted index is detected. Files cover
binary row/column-major, numeric ASCII (both parse paths) and mixed ASCII with and without row counts.
"""

import io

import numpy as np
import pandas as pd
import pytest
import pysdds
from pysdds.structures import Array

N_PAGES = 3
ROWS = [5, 0, 1200]  # middle page is empty; last page is above the small-page threshold of the numeric parser


def _cell(p, k, r):
    return 1000 * p + 10 * k + r


def _make_file(mode, column_major_order, no_row_counts, numeric_only):
    dfs = []
    for p in range(N_PAGES):
        r = np.arange(ROWS[p])
        cols = {
            "c0": _cell(p, 0, r).astype(float),
            "c1": _cell(p, 1, r).astype(np.int32),
            "c2": pd.array([f"s{p}_2_{i}" for i in r], dtype="string"),
            "c3": pd.array([chr(ord("a") + (p * 7 + i) % 26) for i in r], dtype="string"),
            "c4": _cell(p, 4, r).astype(np.int16),
        }
        if numeric_only:
            cols = {k: v for k, v in cols.items() if k in ("c0", "c1", "c4")}
        dfs.append(pd.DataFrame(cols))
    params = {"p0": [float(100 * p) for p in range(N_PAGES)], "p1": [f"pg{p}" for p in range(N_PAGES)]}
    sdds = pysdds.SDDSFile.from_df(dfs, parameter_dict=params, mode=mode)
    if not numeric_only:
        sdds.col("c3").nm["type"] = "character"
    a0 = Array({"name": "a0", "type": "double", "dimensions": 2}, sdds)
    a1 = Array({"name": "a1", "type": "string"}, sdds)
    a2 = Array({"name": "a2", "type": "long"}, sdds)
    for p in range(N_PAGES):
        a0.data.append((np.arange(6, dtype=float) + 100 * p).reshape(2, 3))
        a1.data.append(np.array([f"arr{p}_{i}" for i in range(4)], dtype=object))
        a2.data.append(np.arange(3, dtype=np.int32) + 10 * p)
    sdds.arrays.extend([a0, a1, a2])
    sdds.n_arrays = 3
    sdds.data.nm["column_major_order"] = column_major_order
    sdds.data.nm["no_row_counts"] = no_row_counts
    buf = io.BytesIO()
    pysdds.write(sdds, buf, use_best_settings=False)
    return buf.getvalue()


def _same(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


FILES = [
    ("binary", 0, 0),
    ("binary", 1, 0),
    ("ascii", 0, 0),
    ("ascii", 0, 1),
]
COL_SELECTIONS = [None, ["c4", "c2", "c0"], ["c1"], [1, 3], [4], []]
PAGE_SELECTIONS = [None, [1], [2], [0, 2]]
ARRAY_SELECTIONS = [None, ["a2"], [1], []]


@pytest.mark.parametrize("numeric_only", [False, True])
@pytest.mark.parametrize("mode,column_major_order,no_row_counts", FILES)
def test_selection_matches_full_read(mode, column_major_order, no_row_counts, numeric_only):
    data = _make_file(mode, column_major_order, no_row_counts, numeric_only)
    full = pysdds.read(io.BytesIO(data))
    full.validate_data()
    # The generator is the ground truth for the full read
    for p in range(N_PAGES):
        r = np.arange(ROWS[p])
        assert np.array_equal(full.col("c0").data[p], _cell(p, 0, r).astype(float))
        assert np.array_equal(full.col("c4").data[p], _cell(p, 4, r))
        if not numeric_only:
            assert list(full.col("c2").data[p]) == [f"s{p}_2_{i}" for i in r]
    names = [c.name for c in full.columns]

    for cols in COL_SELECTIONS:
        if cols and isinstance(cols[0], str) and any(c not in names for c in cols):
            continue
        if cols and isinstance(cols[0], int) and max(cols) >= len(names):
            continue
        for pages in PAGE_SELECTIONS:
            for arrays in ARRAY_SELECTIONS:
                sel = pysdds.read(io.BytesIO(data), cols=cols, pages=pages, arrays=arrays)
                ctx = f"cols={cols} pages={pages} arrays={arrays}"
                page_list = list(range(N_PAGES)) if pages is None else pages
                assert sel.n_pages == len(page_list), ctx
                for par in full.parameters:
                    assert sel.par(par.name).data == [par.data[p] for p in page_list], f"{ctx} {par.name}"
                if cols is None:
                    wanted = set(names)
                elif cols and isinstance(cols[0], int):
                    wanted = {names[i] for i in cols}
                else:
                    wanted = set(cols)
                for c in full.columns:
                    got = sel.col(c.name)
                    if c.name not in wanted:
                        assert not got._enabled and got.data == [], f"{ctx} {c.name} should be skipped"
                        continue
                    assert got._enabled and len(got.data) == len(page_list), f"{ctx} {c.name}"
                    for j, p in enumerate(page_list):
                        assert _same(got.data[j], c.data[p]), f"{ctx} {c.name} page {p}"
                if arrays is None:
                    awanted = {a.name for a in full.arrays}
                elif arrays and isinstance(arrays[0], int):
                    awanted = {full.arrays[i].name for i in arrays}
                else:
                    awanted = set(arrays)
                for a in full.arrays:
                    got = sel.array(a.name)
                    if a.name not in awanted:
                        assert got.data == [], f"{ctx} {a.name} should be skipped"
                        continue
                    assert len(got.data) == len(page_list), f"{ctx} {a.name}"
                    for j, p in enumerate(page_list):
                        assert _same(got.data[j], a.data[p]), f"{ctx} {a.name} page {p}"
