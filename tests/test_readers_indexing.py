import numpy as np
import pytest
import pysdds
from pathlib import Path

cwd = Path(__file__).parent
root_sources = cwd / 'files'
to_str = lambda l: [str(s) for s in l]
files_ascii = to_str((root_sources / 'sources_ascii').glob('*'))
get_names = lambda xa: [Path(x).name for x in xa]
get_name = lambda x: Path(x).name


@pytest.mark.parametrize("file_root", files_ascii)
def test_read_ascii(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()
    for c in sdds.columns:
        for p in range(sdds.n_pages):
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
