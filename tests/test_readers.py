import io

import pytest
import pysdds
from pathlib import Path
import itertools

cwd = Path(__file__).parent
root_sources = cwd / "files"
print(f"Executing readers in {cwd=} {root_sources=}")
subroots = ["sources", "sources_binary_rowmajor", "sources_binary_colmajor", "sources_ascii"]


def to_str(l):
    return [str(s) for s in l]


ff = to_str((root_sources / "sources").glob("*"))
ff_ascii = to_str((root_sources / "sources_ascii").glob("*"))
ff_binary_colmajor = to_str((root_sources / "sources_binary_colmajor").glob("*"))
ff_binary_rowmajor = to_str((root_sources / "sources_binary_rowmajor").glob("*"))

fc = to_str((root_sources / "sources_compressed").glob("*"))
fc_ascii = to_str((root_sources / "sources_compressed_ascii").glob("*"))
fc_binary_colmajor = to_str((root_sources / "sources_compressed_binary_colmajor").glob("*"))
fc_binary_rowmajor = to_str((root_sources / "sources_compressed_binary_rowmajor").glob("*"))

fl = to_str((root_sources / "sources_large").glob("*"))
fl_ascii = to_str((root_sources / "sources_large_ascii").glob("*"))
fl_binary_colmajor = to_str((root_sources / "sources_large_binary_colmajor").glob("*"))
fl_binary_rowmajor = to_str((root_sources / "sources_large_binary_rowmajor").glob("*"))

files_all_nolarge = ff + ff_ascii + ff_binary_colmajor + ff_binary_rowmajor + fc
files_all = ff + ff_ascii + ff_binary_colmajor + ff_binary_rowmajor + fl + fc


def get_names(xa):
    return [Path(x).name for x in xa]


def get_name(x):
    return Path(x).name


# Generate pairs [(file1, file1_ascii, ...), (file2, ...)] for equality testing
file_name_dict = {}
for l in [
    ff + fc + fl,
    ff_ascii + fc_ascii + fl_ascii,
    ff_binary_rowmajor + fc_binary_rowmajor + fl_binary_rowmajor,
    ff_binary_colmajor + fc_binary_colmajor + fl_binary_colmajor,
]:
    for f in l:
        name = get_name(f)
        if name in file_name_dict:
            file_name_dict[name].append(f)
        else:
            file_name_dict[name] = [f]

all_tuples = list(file_name_dict.values())

file_name_dict = {}
for l in [ff_binary_rowmajor, ff_binary_colmajor]:
    for f in l:
        name = get_name(f)
        if name in file_name_dict:
            file_name_dict[name].append(f)
        else:
            file_name_dict[name] = [f]

binary_tuples = list(file_name_dict.values())


@pytest.mark.parametrize("file_root", files_all)
def test_read_header(file_root):
    pysdds.read(file_root, header_only=True)


@pytest.mark.parametrize("file_root", ff + fc)
def test_read(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


# TODO: compressed stream support
@pytest.mark.parametrize("file_root", ff)
def test_read_buffer(file_root):
    with open(file_root, "rb") as fs:
        buf = fs.read()
        stream = io.BytesIO(buf)
        bstream = io.BufferedReader(stream)
        sdds = pysdds.read(bstream)
        sdds.validate_data()


@pytest.mark.parametrize("file_root", fl)
def test_read_large(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", ff_binary_colmajor + fc_binary_colmajor)
def test_read_binary1(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", ff_binary_rowmajor + fc_binary_rowmajor)
def test_read_binary2(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", ff_ascii + fc_ascii)
def test_read_ascii(file_root):
    sdds = pysdds.read(file_root)
    sdds.validate_data()
    stream_ascii_windows = open(file_root, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    winstream = io.BytesIO(stream_ascii_windows)
    bstream = io.BufferedReader(winstream)
    sdds = pysdds.read(bstream)
    sdds.validate_data()


@pytest.mark.parametrize("file_root", ff_ascii)
def test_read_ascii_win(file_root):
    stream_ascii_windows = open(file_root, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    winstream = io.BytesIO(stream_ascii_windows)
    bstream = io.BufferedReader(winstream)
    sdds = pysdds.read(bstream)
    sdds.validate_data()


@pytest.mark.parametrize("files", binary_tuples)
def test_read_data_compare_exact(files):
    sdds_objects = [pysdds.read(f) for f in files]
    for pair in itertools.product(sdds_objects, repeat=2):
        assert pair[0].compare(pair[1], raise_error=True, fixed_value_equivalent=True)


@pytest.mark.parametrize("files", all_tuples)
def test_read_data_compare_all(files):
    sdds_objects = [pysdds.read(f) for f in files]
    for pair in itertools.product(sdds_objects, repeat=2):
        assert pair[0].compare(pair[1], eps=1e-5, raise_error=True, fixed_value_equivalent=True)
