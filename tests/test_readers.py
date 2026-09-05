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


@pytest.mark.parametrize("file_root", ff + ff_ascii + ff_binary_rowmajor)
def test_read_raw_bytesio(file_root):
    """A bare io.BytesIO has no peek(); reader must wrap it rather than fail"""
    with open(file_root, "rb") as fs:
        buf = fs.read()
    sdds_stream = pysdds.read(io.BytesIO(buf))
    sdds_stream.validate_data()
    sdds_file = pysdds.read(file_root)
    assert sdds_file.compare(sdds_stream, raise_error=True)


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


def test_read_ascii_character_column_without_string_columns():
    """Character columns are not numeric and must go through the token parser even when no string column exists"""
    src = (
        b"SDDS1\n"
        b"&column name=c, type=character, &end\n"
        b"&column name=x, type=double, &end\n"
        b"&column name=n, type=long, &end\n"
        b"&data mode=ascii, &end\n"
        b"3\n"
        b"a 1.5 1\n"
        b"\\040 2.5 2\n"
        b'\\" 3.5 3\n'
    )
    sdds = pysdds.read(io.BytesIO(src))
    sdds.validate_data()
    assert sdds.n_pages == 1
    assert list(sdds.col("c").data[0]) == ["a", " ", '"']
    assert np.array_equal(sdds.col("x").data[0], np.array([1.5, 2.5, 3.5]))
    assert np.array_equal(sdds.col("n").data[0], np.array([1, 2, 3], dtype=np.int32))


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "mode,column_major_order,no_row_counts",
    [("binary", 0, 0), ("binary", 1, 0), ("ascii", 0, 0), ("ascii", 0, 1)],
)
def test_read_zero_row_page(mode, column_major_order, no_row_counts, mixed):
    """A page with parameters but no rows must not break any of the page parsers"""
    import pandas as pd

    dfs = [
        pd.DataFrame({"x": [1.0, 2.0], "n": np.array([1, 2], dtype=np.int32)}),
        pd.DataFrame({"x": np.array([], dtype=float), "n": np.array([], dtype=np.int32)}),
        pd.DataFrame({"x": [3.0], "n": np.array([3], dtype=np.int32)}),
    ]
    if mixed:
        for df in dfs:
            df["s"] = pd.array(["a"] * len(df), dtype="string")
    sdds = pysdds.SDDSFile.from_df(dfs, parameter_dict={"p": [10, 20, 30]}, mode=mode)
    sdds.data.nm["column_major_order"] = column_major_order
    sdds.data.nm["no_row_counts"] = no_row_counts
    buf = io.BytesIO()
    pysdds.write(sdds, buf, use_best_settings=False)

    sdds2 = pysdds.read(io.BytesIO(buf.getvalue()))
    sdds2.validate_data()
    assert sdds2.n_pages == 3
    assert [len(v) for v in sdds2.col("x").data] == [2, 0, 1]
    assert sdds2.col("x").data[1].dtype == np.float64
    assert sdds2.col("n").data[1].dtype == np.int32
    assert list(sdds2.par("p").data) == [10, 20, 30]
    assert np.array_equal(sdds2.col("x").data[2], [3.0])


def test_read_ascii_trailing_character_column_octal():
    """Character column written as an octal escape in the last position of a row"""
    src = (
        b"SDDS1\n"
        b"&column name=x, type=double, &end\n"
        b"&column name=c, type=character, &end\n"
        b"&data mode=ascii, &end\n"
        b"2\n"
        b"1.5 \\040\n"
        b"2.5 b\n"
    )
    sdds = pysdds.read(io.BytesIO(src))
    assert list(sdds.col("c").data[0]) == [" ", "b"]


@pytest.mark.parametrize("mixed", [False, True])
def test_ascii_text_parameter_escapes_round_trip(mixed):
    """String/character parameters with escapes must decode the same in both ASCII page parsers"""
    import pandas as pd

    df = pd.DataFrame({"x": [1.0, 2.0]})
    if mixed:
        df["s"] = pd.array(["a", "b"], dtype="string")
    text = 'say "hi" ! ok\ttab \\ back'
    sdds = pysdds.SDDSFile.from_df([df], parameter_dict={"p": [text]}, mode="ascii")
    sdds.add_parameter("c", "character", data=["\t"])
    buf = io.BytesIO()
    pysdds.write(sdds, buf)

    sdds2 = pysdds.read(io.BytesIO(buf.getvalue()))
    assert sdds2.par("p").data[0] == text
    assert sdds2.par("c").data[0] == "\t"


def test_read_ascii_no_page_data_terminates():
    """Only fixed parameters and trailing lines: nothing is consumed per page, reader must not spin forever"""
    import signal

    src = (
        b"SDDS1\n&parameter name=p, type=double, fixed_value=1.5, &end\n&data mode=ascii, &end\n\n! trailing comment\n"
    )

    def on_alarm(signum, frame):
        raise TimeoutError("reader hung")

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(10)
    try:
        sdds = pysdds.read(io.BytesIO(src))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert sdds.n_pages == 1
    assert sdds.par("p").data == [1.5]
    # Global pushback buffer must not leak into the next read
    from pysdds.readers.readers import pushback_line_buf

    assert len(pushback_line_buf) == 0


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


def test_read_ascii_numeric_comment_inside_column_data():
    """Comment lines between rows must not count towards the declared row total"""
    src = (
        b"SDDS1\n"
        b"&column name=x, type=double, &end\n"
        b"&data mode=ascii, &end\n"
        b"2\n"
        b"1.0\n"
        b"! comment between rows\n"
        b"2.0 ! trailing comment\n"
        b"2\n"
        b"3.0\n"
        b"4.0\n"
    )
    sdds = pysdds.read(io.BytesIO(src))
    assert sdds.n_pages == 2
    assert [v.tolist() for v in sdds.col("x").data] == [[1.0, 2.0], [3.0, 4.0]]


def test_read_ascii_numeric_no_row_counts_crlf():
    """Empty-line page terminator must be recognised with Windows line endings in the numeric parser"""
    src = (
        b"SDDS1\r\n"
        b"&parameter name=p, type=long, &end\r\n"
        b"&column name=x, type=double, &end\r\n"
        b"&data mode=ascii, no_row_counts=1, &end\r\n"
        b"1\r\n1.0\r\n2.0\r\n"
        b"\r\n"
        b"2\r\n3.0\r\n"
    )
    sdds = pysdds.read(io.BytesIO(src))
    assert sdds.n_pages == 2
    assert [v.tolist() for v in sdds.col("x").data] == [[1.0, 2.0], [3.0]]
    assert sdds.par("p").data == [1, 2]


def test_read_header_whitespace_around_equals():
    src = b'SDDS1\n&column name = x, type =double, description= "a b", &end\n&data mode = ascii, &end\n1\n1.0\n'
    sdds = pysdds.read(io.BytesIO(src))
    assert sdds.column_names == ["x"]
    assert sdds.col("x").nm == {"name": "x", "type": "double", "description": "a b"}
    assert sdds.col("x").data[0].tolist() == [1.0]


def test_read_header_without_data_namelist():
    """&data is optional when the file has only a description and fixed-value parameters"""
    src = b'SDDS1\n&description text="header only", &end\n&parameter name=p, type=double, fixed_value=1.5, &end\n'
    sdds = pysdds.read(io.BytesIO(src))
    assert sdds.data is None
    assert sdds.n_pages == 0
    assert sdds.par("p").fixed_value == 1.5

    with pytest.raises(pysdds.util.errors.SDDSReadError):
        pysdds.read(io.BytesIO(src + b"&column name=x, type=double, &end\n"))
