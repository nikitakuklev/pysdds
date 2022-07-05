import numpy as np
import pytest
import pysdds
import glob
from pathlib import Path
import itertools

root_ascii = 'files/ascii/'
files_ascii = glob.glob(root_ascii + '*')

get_names = lambda xa: [Path(x).name for x in xa]
get_name = lambda x: Path(x).name

@pytest.mark.parametrize("file_root", files_ascii)
def test_read_ascii(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()
    for c in sdds.columns:
        for p in sdds.n_pages:
            col = sdds.col(c.name)
            a = col[p]
            b = col.data[p]
            assert np.array_equal(a, b)


@pytest.mark.parametrize("file_root", ['files/sources/timeSeries.sdds'])
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

