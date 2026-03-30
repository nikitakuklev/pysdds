import io

import pytest
import pysdds
from pathlib import Path
import itertools
import pandas as pd
import numpy as np

cwd = Path(__file__).parent
root_sources = cwd / "files"
root_binary_rowmajor = root_sources / "sources_binary_rowmajor"
root_binary_colmajor = root_sources / "sources_binary_colmajor"
root_ascii = root_sources / "ascii"


def to_str(slist):
    return [str(s) for s in slist]


files_sources = to_str((root_sources / "sources").glob("*"))
files_ascii = to_str((root_sources / "sources_ascii").glob("*"))
files_binary_colmajor = to_str((root_sources / "sources_binary_colmajor").glob("*"))
files_binary_rowmajor = to_str((root_sources / "sources_binary_rowmajor").glob("*"))
files_compressed = to_str((root_sources / "sources_compressed").glob("*"))
files_large = to_str((root_sources / "sources_large").glob("*"))

all_files = files_sources + files_ascii + files_binary_colmajor + files_binary_rowmajor + files_large


def get_names(xa):
    return [Path(x).name for x in xa]


def get_name(x):
    return Path(x).name


# Generate pairs [(file1, file1_ascii, ...), (file2, ...)] for equality testing
set_sources = set(get_names(files_sources))
set_ascii_rowmajor = set(get_names(files_ascii))
set_binary_rowmajor = set(get_names(files_binary_rowmajor))
set_binary_colmajor = set(get_names(files_binary_colmajor))

set_union = set_sources.intersection(set_ascii_rowmajor, set_binary_rowmajor, set_binary_colmajor)
sets_list = []
for f in set_sources:
    if f in set_union:
        sets_list.append(
            [str(root_sources) + f, str(root_binary_colmajor) + f, str(root_binary_rowmajor) + f, str(root_ascii) + f]
        )
    else:
        continue

# This will create AB and BA comparison to ensure things are commutative
all_tuples = [x for files in sets_list for x in itertools.product(files, repeat=2)]
all_files1 = [x[0] for x in all_tuples]
all_files2 = [x[1] for x in all_tuples]


@pytest.mark.parametrize("file_root", files_ascii)
def test_round_trip_ascii_ascii(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    # lines = buf.getvalue().decode('ascii').split('\n')
    # for l in lines:
    #     print(l)
    #     pass
    # print('--')
    sdds2 = pysdds.read(io.BufferedReader(buf))
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    # print(buf.getvalue().decode())
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == sdds.endianness
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_ascii(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode("ascii")
    pysdds.write(sdds, buf)
    buf.seek(0)
    #
    # print()
    # print("=====================================")
    # lines = buf.getvalue().decode('ascii').split('\n')
    # for l in lines:
    #     print(repr(l))
    #     pass
    # print("=====================================")

    # with open(file_root+'_writeback', "wb") as f:
    #     f.write(buf.getbuffer())

    sdds2 = pysdds.read(io.BufferedReader(buf))
    sdds.compare(sdds2)

    # sdds3 = pysdds.read(file_root+'_writeback')
    # sdds.compare(sdds3)
    # sdds2.compare(sdds3)
    # os.remove(os.path.abspath(file_root+'_writeback'))


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_bincol_le(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode("binary")
    sdds.set_endianness("little")
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == "little"
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_bincol_be(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode("binary")
    sdds.set_endianness("big")
    pysdds.write(sdds, buf)
    buf.seek(0)
    # lines = buf.getvalue()
    # print(lines)
    # print('--')
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == "big"
    sdds.compare(sdds2)


def test_write_from_df():
    meas_df = {
        "ControlName": ["foo-_~|aaaa", "bar\n\r\005"],
        "LowerLimit": [-2, -2],
        "UpperLimit": [-2.0, -2.0],
        "UnsignedVal": [+3, +4],
        "Description": [
            "foo-_~|aaaa",
            "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
        ],
    }
    df_meas = pd.DataFrame.from_dict(meas_df)
    df_meas["ReallyUnsigned"] = np.array([4, 4], dtype=np.uint64)
    df_meas['Horror!@#$)(*&^%#@&$#&$^@}{":?<'] = np.array([5.0, 5.1], dtype=np.float32)
    parameters = {"par1": [1], "par2": [1.0], "par3": ["foo"]}

    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode="ascii")
    assert sdds.n_pages == 1
    assert sdds.n_columns == 7
    assert sdds.n_parameters == 3
    assert sdds.columns[0].type == "string"
    assert sdds.columns[1].type == "long64"
    assert sdds.columns[2].type == "double"
    assert sdds.columns[3].type == "long64"
    assert sdds.columns[4].type == "string"
    assert sdds.columns[5].type == "ulong64"
    assert sdds.columns[6].type == "float"
    assert sdds.columns[0].data[0].dtype == object
    assert sdds.columns[1].data[0].dtype == np.int64
    assert sdds.columns[2].data[0].dtype == np.float64
    assert sdds.columns[3].data[0].dtype == np.int64
    assert sdds.columns[4].data[0].dtype == object
    assert sdds.columns[5].data[0].dtype == np.uint64
    assert sdds.columns[6].data[0].dtype == np.float32

    def run_compare(sdds):
        sdds.validate_data()
        buf = io.BytesIO()
        pysdds.write(sdds, buf)
        buf.seek(0)
        sdds2 = pysdds.read(io.BufferedReader(buf))
        sdds.compare(sdds2)
        assert np.array_equal(sdds2.columns[0].data[0], df_meas.iloc[:, 0])
        assert np.array_equal(sdds2.columns[1].data[0], df_meas.iloc[:, 1])
        assert np.array_equal(sdds2.columns[2].data[0], df_meas.iloc[:, 2])
        assert np.array_equal(sdds2.columns[3].data[0], df_meas.iloc[:, 3])
        assert np.array_equal(sdds2.columns[4].data[0], df_meas.iloc[:, 4])

    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode="ascii")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], mode="ascii")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode="binary")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], mode="binary")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], mode="binary", endianness="big")
    run_compare(sdds)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_best_settings_column_major(file_root):
    """Test that use_best_settings=True (column-major) round-trips correctly."""
    sdds = pysdds.read(file_root)
    sdds.set_mode("binary")
    buf = io.BytesIO()
    pysdds.write(sdds, buf, use_best_settings=True)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    sdds.compare(sdds2)


# ---------------------------------------------------------------------------
# Column-major vs row-major writer tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file_root", files_sources)
def test_write_binary_row_major_round_trip(file_root):
    """Write binary row-major, read back, compare data."""
    sdds = pysdds.read(file_root)
    sdds.set_mode("binary")
    sdds.set_endianness("little")
    sdds.data.nm["column_major_order"] = 0
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.data.column_major_order == 0
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_write_binary_col_major_round_trip(file_root):
    """Write binary column-major, read back, compare data."""
    sdds = pysdds.read(file_root)
    sdds.set_mode("binary")
    sdds.set_endianness("little")
    sdds.data.nm["column_major_order"] = 1
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.data.column_major_order == 1
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_row_major_and_col_major_produce_same_data(file_root):
    """Row-major and column-major writes must produce identical data when read back."""
    sdds = pysdds.read(file_root)
    sdds.set_mode("binary")
    sdds.set_endianness("little")

    sdds.data.nm["column_major_order"] = 0
    buf_row = io.BytesIO()
    pysdds.write(sdds, buf_row)
    buf_row.seek(0)
    sdds_row = pysdds.read(io.BufferedReader(buf_row))

    sdds.data.nm["column_major_order"] = 1
    buf_col = io.BytesIO()
    pysdds.write(sdds, buf_col)
    buf_col.seek(0)
    sdds_col = pysdds.read(io.BufferedReader(buf_col))

    sdds_row.compare(sdds_col)


@pytest.mark.parametrize("file_root", files_sources)
def test_col_major_big_endian_round_trip(file_root):
    """Column-major + big-endian round-trip."""
    sdds = pysdds.read(file_root)
    sdds.set_mode("binary")
    sdds.set_endianness("big")
    sdds.data.nm["column_major_order"] = 1
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == "big"
    assert sdds2.data.column_major_order == 1
    sdds.compare(sdds2)


# ---------------------------------------------------------------------------
# Byte-level comparison: pysdds output vs C sddsconvert reference files
# ---------------------------------------------------------------------------


def _skip_header(raw: bytes) -> int:
    """Return the byte offset just past the header (after the &data...&end line)."""
    # Header is ASCII text ending with a line containing "&data ... &end"
    # followed by a newline, then binary data begins.
    pos = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        if nl == -1:
            break
        line = raw[pos:nl]
        if b"&data" in line and b"&end" in line:
            return nl + 1
        pos = nl + 1
    raise ValueError("Could not find &data ... &end in header")


# Files that exist in both sources/ and sources_binary_colmajor/ (generated by C sddsconvert)
_ref_colmajor_pairs = [
    (str(root_sources / "sources" / name), str(root_binary_colmajor / name))
    for name in sorted(set_sources & set_binary_colmajor)
]
_ref_rowmajor_pairs = [
    (str(root_sources / "sources" / name), str(root_binary_rowmajor / name))
    for name in sorted(set_sources & set_binary_rowmajor)
]


@pytest.mark.parametrize("source,ref_file", _ref_rowmajor_pairs, ids=[Path(p[0]).name for p in _ref_rowmajor_pairs])
def test_binary_data_matches_c_reference_rowmajor(source, ref_file):
    """Read the C sddsconvert-produced reference, rewrite as binary row-major LE,
    compare data sections byte-for-byte. Uses the reference as input so both
    sides agree on fixed-value parameter handling."""
    ref_sdds = pysdds.read(ref_file)

    buf = io.BytesIO()
    pysdds.write(ref_sdds, buf)
    py_bytes = buf.getvalue()

    with open(ref_file, "rb") as f:
        ref_bytes = f.read()

    # Compare data sections (skip headers since comments/formatting may differ)
    py_data = py_bytes[_skip_header(py_bytes) :]
    ref_data = ref_bytes[_skip_header(ref_bytes) :]
    assert py_data == ref_data, (
        f"Binary data mismatch: pysdds wrote {len(py_data)} bytes, "
        f"reference has {len(ref_data)} bytes. "
        f"First diff at byte {next((i for i in range(min(len(py_data), len(ref_data))) if py_data[i] != ref_data[i]), -1)}"
    )


@pytest.mark.parametrize("source,ref_file", _ref_colmajor_pairs, ids=[Path(p[0]).name for p in _ref_colmajor_pairs])
def test_binary_data_matches_c_reference_colmajor(source, ref_file):
    """Read the C sddsconvert-produced reference, rewrite as binary col-major LE,
    compare data sections byte-for-byte."""
    ref_sdds = pysdds.read(ref_file)

    buf = io.BytesIO()
    pysdds.write(ref_sdds, buf)
    py_bytes = buf.getvalue()

    with open(ref_file, "rb") as f:
        ref_bytes = f.read()

    py_data = py_bytes[_skip_header(py_bytes) :]
    ref_data = ref_bytes[_skip_header(ref_bytes) :]
    assert py_data == ref_data, (
        f"Binary data mismatch: pysdds wrote {len(py_data)} bytes, "
        f"reference has {len(ref_data)} bytes. "
        f"First diff at byte {next((i for i in range(min(len(py_data), len(ref_data))) if py_data[i] != ref_data[i]), -1)}"
    )


@pytest.mark.parametrize("source,ref_file", _ref_rowmajor_pairs, ids=[Path(p[0]).name for p in _ref_rowmajor_pairs])
def test_source_to_binary_logical_match_rowmajor(source, ref_file):
    """Read original source, write as binary row-major, read back, compare
    logically against C sddsconvert reference."""
    sdds = pysdds.read(source)
    ref_sdds = pysdds.read(ref_file)

    sdds.set_mode("binary")
    sdds.set_endianness("little")
    sdds.data.nm["column_major_order"] = 0

    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))

    # Logical comparison: our re-read must match the C reference
    ref_sdds.compare(sdds2, ignore_data_mode=True)


@pytest.mark.parametrize("source,ref_file", _ref_colmajor_pairs, ids=[Path(p[0]).name for p in _ref_colmajor_pairs])
def test_source_to_binary_logical_match_colmajor(source, ref_file):
    """Read original source, write as binary col-major, read back, compare
    logically against C sddsconvert reference."""
    sdds = pysdds.read(source)
    ref_sdds = pysdds.read(ref_file)

    sdds.set_mode("binary")
    sdds.set_endianness("little")
    sdds.data.nm["column_major_order"] = 1

    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))

    ref_sdds.compare(sdds2, ignore_data_mode=True)
