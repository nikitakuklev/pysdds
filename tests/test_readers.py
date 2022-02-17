import pytest
import pysdds
import glob
from pathlib import Path
import itertools

root_sources = 'files/sources/'
root_binary_rowmajor = 'files/binary_rowmajor/'
root_binary_colmajor = 'files/binary_colmajor/'
root_ascii = 'files/ascii/'

files_sources = glob.glob(root_sources + '*')
files_ascii = glob.glob(root_ascii + '*')
files_binary_colmajor = glob.glob(root_binary_colmajor+'*')
files_binary_rowmajor = glob.glob(root_binary_rowmajor+'*')
files_compressed = glob.glob('files/sources_compressed/*')
files_large = glob.glob('files/sources_large/*')

all_files = files_sources + files_ascii + files_binary_colmajor + files_binary_rowmajor + files_large

get_names = lambda xa: [Path(x).name for x in xa]
get_name = lambda x: Path(x).name

# Generate pairs [(file1, file1_ascii, ...), (file2, ...)] for equality testing
set_sources = set(get_names(files_sources))
set_ascii_rowmajor = set(get_names(files_ascii))
set_binary_rowmajor = set(get_names(files_binary_rowmajor))
set_binary_colmajor = set(get_names(files_binary_colmajor))

set_union = set_sources.intersection(set_ascii_rowmajor).intersection(set_binary_rowmajor).intersection(set_binary_colmajor)
sets_list = []
for f in set_sources:
    if f in set_union:
        sets_list.append([root_sources + f, root_binary_colmajor + f, root_binary_rowmajor + f, root_ascii + f])
    else:
        continue

# This will create AB and BA comparison to ensure things are commutative
all_tuples = [x for files in sets_list for x in itertools.product(files, repeat=2)]
all_files1 = [x[0] for x in all_tuples]
all_files2 = [x[1] for x in all_tuples]


@pytest.mark.parametrize("file_root", files_sources)
def test_read_header(file_root):
    pysdds.read(file_root, header_only=True)


@pytest.mark.parametrize("file_root", files_sources)
def test_read(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", files_binary_colmajor)
def test_read_binary1(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", files_binary_rowmajor)
def test_read_binary2(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", files_ascii)
def test_read_ascii(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", files_compressed)
def test_read_compressed(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", files_large)
def test_read_large(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


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


@pytest.mark.parametrize("file1, file2", all_tuples)
def test_read_data_compare(file1, file2):
    sdds1 = pysdds.read(file1)
    sdds2 = pysdds.read(file2)
    assert sdds1.compare(sdds2, eps=1e-5, raise_error=True, fixed_value_equivalent=True)