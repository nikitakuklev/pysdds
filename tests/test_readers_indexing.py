import itertools
import logging
import random
import time

import numpy as np
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

    sdds = pysdds.read(file_root, cols=[], pages=[1,3])
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
    logging.basicConfig(level=logging.DEBUG,
                        # format='%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s', #[%(name)s]
                        format='[%(levelname)-5.5s][%(asctime)s.%(msecs)03d '
                               '%(filename)10s %(lineno)4s] %(message)s',
                        datefmt='%H:%M:%S',
                        force=True
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
