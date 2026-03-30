import io

import pytest
import numpy as np
import pysdds
from pathlib import Path
import itertools

cwd = Path(__file__).parent
root_sources = cwd / "files"
print(f"Executing readers in {cwd=} {root_sources=}")
subroots = ["sources", "sources_binary_rowmajor", "sources_binary_colmajor", "sources_ascii"]


def to_str(slist):
    return [str(s) for s in slist]


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
for slist in [
    ff + fc + fl,
    ff_ascii + fc_ascii + fl_ascii,
    ff_binary_rowmajor + fc_binary_rowmajor + fl_binary_rowmajor,
    ff_binary_colmajor + fc_binary_colmajor + fl_binary_colmajor,
]:
    for f in slist:
        name = get_name(f)
        if name in file_name_dict:
            file_name_dict[name].append(f)
        else:
            file_name_dict[name] = [f]

all_tuples = list(file_name_dict.values())

file_name_dict = {}
for slist in [ff_binary_rowmajor, ff_binary_colmajor]:
    for f in slist:
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


def test_masked_string_array_does_not_corrupt_columns():
    """Masking out a string array must still read columns correctly."""
    import io

    source = str(root_sources / "sources" / "L3_QM1.excitation.proc")
    # Read full file as reference
    sdds_full = pysdds.read(source)
    assert len(sdds_full.arrays) == 3
    assert sdds_full.arrays[2].type == "string"

    # Read again, masking out all arrays
    sdds_no_arrays = pysdds.read(source, arrays=[])
    # Columns must still be intact
    for i, col in enumerate(sdds_full.columns):
        assert np.array_equal(col.data[0], sdds_no_arrays.columns[i].data[0]), (
            f"Column {col.name} data mismatch when arrays are masked out"
        )

    # Read again, masking out only the string array (keep numeric arrays)
    sdds_partial = pysdds.read(source, arrays=["Order", "Coefficient"])
    for i, col in enumerate(sdds_full.columns):
        assert np.array_equal(col.data[0], sdds_partial.columns[i].data[0]), (
            f"Column {col.name} data mismatch when string array is masked out"
        )

    # Also verify via round-trip: write to buffer, read back with mask
    buf = io.BytesIO()
    pysdds.write(sdds_full, buf)
    buf.seek(0)
    sdds_rt = pysdds.read(io.BufferedReader(buf), arrays=[])
    for i, col in enumerate(sdds_full.columns):
        assert np.array_equal(col.data[0], sdds_rt.columns[i].data[0]), (
            f"Column {col.name} data mismatch after round-trip with masked arrays"
        )


@pytest.mark.skipif(
    np.dtype(np.longdouble) == np.dtype(np.float64),
    reason="longdouble == float64 on this platform (e.g. Windows), cannot parse 80-bit floats",
)
def test_read_all_sdds_types():
    """Read the reference example.sdds that exercises every SDDS data type:
    short, ushort, long, ulong, long64, ulong64, float, double, longdouble,
    string, character — in parameters, 1D/2D arrays, and columns."""
    source = str(root_sources / "example_all_types.sdds")
    sdds = pysdds.read(source, allow_longdouble=True)

    assert sdds.n_pages == 2
    assert len(sdds.parameters) == 11
    assert len(sdds.arrays) == 11
    assert len(sdds.columns) == 11

    # Verify parameter types and page 1 values
    expected_params = {
        "shortParam": ("short", np.int16(10)),
        "ushortParam": ("ushort", np.uint16(11)),
        "longParam": ("long", np.int32(1000)),
        "ulongParam": ("ulong", np.uint32(1001)),
        "long64Param": ("long64", np.int64(1002)),
        "ulong64Param": ("ulong64", np.uint64(1003)),
        "floatParam": ("float", np.float32(3.14)),
        "doubleParam": ("double", np.float64(2.71828)),
        "stringParam": ("string", "FirstPage"),
        "charParam": ("character", "A"),
    }
    for p in sdds.parameters:
        if p.name in expected_params:
            exp_type, exp_val = expected_params[p.name]
            assert p.type == exp_type, f"{p.name}: type {p.type} != {exp_type}"
            if p.type in ("string", "character"):
                assert p.data[0] == exp_val, f"{p.name}: {p.data[0]} != {exp_val}"
            elif p.type == "float":
                assert np.isclose(p.data[0], exp_val, rtol=1e-5), f"{p.name}: {p.data[0]} != {exp_val}"
            else:
                assert p.data[0] == exp_val, f"{p.name}: {p.data[0]} != {exp_val}"

    # Verify 1D arrays (first 4)
    assert sdds.arrays[0].name == "shortArray"
    assert np.array_equal(sdds.arrays[0].data[0], np.array([1, 2, 3], dtype=np.int16))

    # Verify 2D arrays
    long64_arr = sdds.arrays[4]
    assert long64_arr.name == "long64Array"
    assert long64_arr.dimensions == 2
    assert long64_arr.data[0].shape == (4, 2)
    assert long64_arr.data[0][0, 0] == 1002

    string_arr = sdds.arrays[9]
    assert string_arr.name == "stringArray"
    assert string_arr.data[0].shape == (4, 2)
    assert string_arr.data[0][0, 0] == "one"
    assert string_arr.data[0][3, 1] == "eight"

    char_arr = sdds.arrays[10]
    assert char_arr.name == "charArray"
    assert char_arr.data[0][0, 0] == "A"

    # Verify column data page 1
    assert np.array_equal(sdds.columns[0].data[0], np.array([1, 2, 3, 4, 5], dtype=np.int16))
    assert list(sdds.columns[9].data[0]) == ["one", "two", "three", "four", "five"]
    assert list(sdds.columns[10].data[0]) == ["a", "b", "c", "d", "e"]

    # Verify page 2
    assert sdds.parameters[0].data[1] == np.int16(20)
    assert list(sdds.columns[9].data[1]) == ["six", "seven", "eight"]


def test_read_all_sdds_types_header_only():
    """Verify header parsing of all SDDS types works on every platform
    (no longdouble data is actually parsed, just the header)."""
    source = str(root_sources / "example_all_types.sdds")
    sdds = pysdds.read(source, header_only=True)

    assert len(sdds.parameters) == 11
    assert len(sdds.arrays) == 11
    assert len(sdds.columns) == 11

    expected_types = [
        "short",
        "ushort",
        "long",
        "ulong",
        "long64",
        "ulong64",
        "float",
        "double",
        "longdouble",
        "string",
        "character",
    ]
    assert [p.type for p in sdds.parameters] == expected_types
    assert [a.type for a in sdds.arrays] == expected_types
    assert [c.type for c in sdds.columns] == expected_types

    # Verify multi-dimensional array declarations
    for a in sdds.arrays[:4]:
        assert a.dimensions == 1
    for a in sdds.arrays[4:]:
        assert a.dimensions == 2
