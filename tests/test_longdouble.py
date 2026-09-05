"""longdouble handling on every platform.

SDDS files carry longdouble as 16-byte x87 extended fields. On x86-64 Linux numpy reads those directly as
np.longdouble; on Windows and Apple Silicon np.longdouble is plain float64, so pysdds decodes and encodes the
fields itself. Binary fixtures here are built from hardcoded field bytes so the tests do not depend on the host.
"""

import io
import struct
import warnings

import numpy as np
import pytest

import pysdds
from pysdds.util.constants import _LONGDOUBLE_NATIVE as NATIVE
from pysdds.util.constants import _NUMPY_DTYPE_FINAL
from pysdds.util.conversions import decode_longdouble_fields, encode_longdouble_fields

LD = _NUMPY_DTYPE_FINAL["longdouble"]  # 16-byte extended where native, float64 otherwise

# value -> low 10 bytes of the little-endian field (written by numpy on x86-64 Linux)
FIELDS = {
    "1.000000000000000001": bytes.fromhex("0900000000000080ff3f"),
    "2.5": bytes.fromhex("00000000000000a00040"),
    "-3.25": bytes.fromhex("00000000000000d000c0"),
    "1.0": bytes.fromhex("0000000000000080ff3f"),
    "0": bytes(10),
    "1e-4940": bytes.fromhex("628e2763060000000000"),  # far below double range
    "1e4000": bytes.fromhex("618c55fe2383bad1e673"),  # far above double range
}
VALUES = ["1.000000000000000001", "2.5", "-3.25"]


def _field(value: str, pad: bytes = b"\xaa" * 6) -> bytes:
    """A full 16-byte field; the C library leaves the padding uninitialised so use non-zero junk"""
    return FIELDS[value] + pad


def _expected(values):
    """What a read of these values should produce on this platform"""
    return np.array([np.longdouble(v) for v in values], dtype=LD)


def test_decode_known_fields():
    buf = b"".join(_field(v) for v in FIELDS)
    out = decode_longdouble_fields(buf)
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.5, -3.25, 1.0, 0.0, 0.0, np.inf]


def test_decode_specials():
    inf = bytes.fromhex("0000000000000080ff7f") + bytes(6)
    neg_inf = bytes.fromhex("0000000000000080ffff") + bytes(6)
    nan = bytes.fromhex("00000000000000c0ff7f") + bytes(6)
    neg_zero = bytes.fromhex("00000000000000000080") + bytes(6)
    out = decode_longdouble_fields(inf + neg_inf + nan + neg_zero)
    assert out[0] == np.inf and out[1] == -np.inf and np.isnan(out[2])
    assert out[3] == 0.0 and np.signbit(out[3])


def test_encode_known_fields():
    packed = encode_longdouble_fields([1.0, 2.5, -3.25, 0.0])
    fields = [packed[i * 16 : (i + 1) * 16] for i in range(4)]
    assert fields[0] == FIELDS["1.0"] + bytes(6)
    assert fields[1] == FIELDS["2.5"] + bytes(6)
    assert fields[2] == FIELDS["-3.25"] + bytes(6)
    assert fields[3] == bytes(16)


def test_codec_round_trip_doubles():
    rng = np.random.default_rng(1)
    vals = np.concatenate(
        [
            rng.standard_normal(20000) * 10.0 ** rng.integers(-300, 300, 20000),
            [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308],
        ]
    )
    back = decode_longdouble_fields(encode_longdouble_fields(vals))
    assert np.array_equal(back, vals, equal_nan=True)
    assert np.array_equal(np.signbit(back), np.signbit(vals))


@pytest.mark.skipif(not NATIVE, reason="needs the 16-byte np.longdouble to compare against")
def test_codec_matches_native_longdouble():
    rng = np.random.default_rng(2)
    vals = rng.standard_normal(20000) * 10.0 ** rng.integers(-300, 300, 20000)
    native = vals.astype(np.longdouble)
    assert np.array_equal(decode_longdouble_fields(native.tobytes()), vals)
    # numpy's own field bytes and ours agree (padding aside)
    ours = np.frombuffer(encode_longdouble_fields(vals), dtype=np.uint8).reshape(-1, 16)[:, :10]
    theirs = np.frombuffer(native.tobytes(), dtype=np.uint8).reshape(-1, 16)[:, :10]
    assert np.array_equal(ours, theirs)


def test_decode_rejects_big_endian_and_bad_length():
    with pytest.raises(NotImplementedError):
        decode_longdouble_fields(bytes(16), "big")
    with pytest.raises(ValueError):
        decode_longdouble_fields(bytes(15))


def _binary_fixture(column_major: bool, with_string: bool, fixed_rowcount: bool = False) -> bytes:
    """One page: longdouble parameter, 1D longdouble array, long + longdouble (+ string) columns, 3 rows"""
    header = b"SDDS1\n"
    if fixed_rowcount:
        header += b"!# fixed-rowcount\n"
    header += (
        b"&parameter name=pg, type=longdouble, &end\n"
        b"&array name=ag, type=longdouble, &end\n"
        b"&column name=x, type=long, &end\n"
        b"&column name=g, type=longdouble, &end\n"
    )
    if with_string:
        header += b"&column name=s, type=string, &end\n"
    header += b"&data mode=binary, column_major_order=%d, &end\n" % int(column_major)
    strings = [b"a", b"bb", b"ccc"]
    body = struct.pack("<i", 3)
    body += _field(VALUES[0])
    body += struct.pack("<i", 3) + b"".join(_field(v) for v in VALUES)
    if column_major:
        body += struct.pack("<3i", 0, 1, 2)
        body += b"".join(_field(v) for v in VALUES)
        if with_string:
            body += b"".join(struct.pack("<i", len(s)) + s for s in strings)
    else:
        for row in range(3):
            body += struct.pack("<i", row) + _field(VALUES[row])
            if with_string:
                body += struct.pack("<i", len(strings[row])) + strings[row]
    return header + body


@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("with_string", [False, True])
@pytest.mark.parametrize("fixed_rowcount", [False, True])
def test_read_binary_longdouble(column_major, with_string, fixed_rowcount):
    if fixed_rowcount and column_major:
        pytest.skip("fixed rowcount is a row-major mode")
    src = _binary_fixture(column_major, with_string, fixed_rowcount)
    sdds = pysdds.read(io.BytesIO(src), allow_longdouble=True)
    expected = _expected(VALUES)
    g = sdds.col("g").data[0]
    assert g.dtype == LD
    assert np.array_equal(g, expected)
    assert np.array_equal(sdds.arrays[0].data[0], expected)
    assert sdds.par("pg").data[0] == expected[0]
    assert sdds.col("x").data[0].tolist() == [0, 1, 2]
    if with_string:
        assert sdds.col("s").data[0].tolist() == ["a", "bb", "ccc"]
    if NATIVE:
        assert g[0] != np.longdouble(1.0), "80-bit precision must survive where the platform has it"
    else:
        assert g[0] == 1.0


def test_read_binary_longdouble_without_permission():
    with pytest.raises(ValueError, match="allow_longdouble"):
        pysdds.read(io.BytesIO(_binary_fixture(False, False)))


def test_read_binary_longdouble_out_of_double_range():
    src = (
        _binary_fixture(True, False)
        .replace(_field(VALUES[1]), _field("1e4000"))
        .replace(_field(VALUES[2]), _field("1e-4940"))
    )
    sdds = pysdds.read(io.BytesIO(src), allow_longdouble=True)
    g = sdds.col("g").data[0]
    if NATIVE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # numpy warns while parsing these, but parses them
            big, tiny = np.longdouble("1e4000"), np.longdouble("1e-4940")
        assert g[1] == big and g[2] == tiny
    else:
        assert g[1] == np.inf and g[2] == 0.0


def _build(mode, with_string, column_major, endianness="little", with_data=True):
    values = _expected(VALUES)
    sdds = pysdds.SDDSFile(add_data_nm=True)
    sdds.set_mode(mode)
    sdds.set_endianness(endianness)
    sdds.data.nm["column_major_order"] = int(column_major)
    sdds.add_parameter("pg", "longdouble", data=[values[0]] if with_data else None)
    sdds.arrays.append(pysdds.structures.Array({"name": "ag", "type": "longdouble"}, sdds))
    sdds.n_arrays = 1
    if with_data:
        sdds.arrays[0].data = [values.copy()]
    sdds.add_column("x", "long", data=[np.arange(3, dtype=np.int32)] if with_data else None)
    sdds.add_column("g", "longdouble", data=[values.copy()] if with_data else None)
    if with_string:
        sdds.add_column("s", "string", data=[np.array(["a", "bb", "ccc"], dtype=object)] if with_data else None)
    if with_data:
        sdds.n_pages = 1
    return sdds


def _check_round_trip(sdds2):
    expected = _expected(VALUES)
    g = sdds2.col("g").data[0]
    assert g.dtype == LD
    assert np.array_equal(g, expected)
    assert np.array_equal(sdds2.arrays[0].data[0], expected)
    assert sdds2.par("pg").data[0] == expected[0]
    assert sdds2.col("x").data[0].tolist() == [0, 1, 2]


@pytest.mark.parametrize("mode", ["binary", "ascii"])
@pytest.mark.parametrize("with_string", [False, True])
@pytest.mark.parametrize("column_major", [False, True])
def test_write_read_longdouble_round_trip(mode, with_string, column_major):
    if mode == "ascii" and column_major:
        pytest.skip("column order does not apply to ascii")
    buf = io.BytesIO()
    pysdds.write(_build(mode, with_string, column_major), buf)
    buf.seek(0)
    _check_round_trip(pysdds.read(io.BufferedReader(buf), allow_longdouble=True))


@pytest.mark.parametrize("with_string", [False, True])
@pytest.mark.parametrize("n_rows_declared", [3, 5])
def test_streaming_writer_longdouble(with_string, n_rows_declared):
    values = _expected(VALUES)
    sdds = _build("binary", with_string, False, with_data=False)
    buf = io.BytesIO()
    w = sdds.get_streaming_writer(buf)
    w.binary_fixed_rowcount = n_rows_declared
    w.begin()
    w.new_page([values[0]], [values.copy()])
    cols = [np.arange(3, dtype=np.int32), values.copy()]
    if with_string:
        cols.append(np.array(["a", "bb", "ccc"], dtype=object))
    w.write_rows(cols)
    w.close()
    _check_round_trip(pysdds.read(io.BytesIO(buf.getvalue()), allow_longdouble=True))


def test_streaming_writer_longdouble_single_row():
    sdds = _build("binary", False, False, with_data=False)
    buf = io.BytesIO()
    w = sdds.get_streaming_writer(buf)
    w.binary_fixed_rowcount = 1
    w.begin()
    w.new_page([2.5], [np.array([2.5], dtype=LD)])
    w.write_rows([1, 2.5])
    w.close()
    sdds2 = pysdds.read(io.BytesIO(buf.getvalue()), allow_longdouble=True)
    assert sdds2.col("g").data[0].tolist() == [2.5]


@pytest.mark.skipif(NATIVE, reason="only platforms without an 80-bit type refuse big-endian longdouble")
def test_big_endian_longdouble_refused_without_native_type():
    with pytest.raises(NotImplementedError):
        buf = io.BytesIO()
        pysdds.write(_build("binary", False, False, endianness="big"), buf)


@pytest.mark.parametrize("with_string", [False, True])
def test_read_ascii_longdouble_large_page_keeps_precision(with_string):
    """Pages above the small-page threshold go through pandas, which must not round longdouble to double"""
    n = 1500
    extra_col = b"&column name=s, type=string, &end\n" if with_string else b""
    extra_val = " q" if with_string else ""
    rows = "\n".join(f"{i} 1.00000000000000000{i % 10}e+00{extra_val}" for i in range(n))
    src = (
        b"SDDS1\n&column name=i, type=long, &end\n&column name=g, type=longdouble, &end\n"
        + extra_col
        + b"&data mode=ascii, &end\n"
        + f"{n}\n".encode()
        + rows.encode()
        + b"\n"
    )
    sdds = pysdds.read(io.BytesIO(src), allow_longdouble=True)
    g = sdds.col("g").data[0]
    assert g.dtype == LD and len(g) == n
    assert g[0] == 1.0
    assert g[3] == LD.type("1.000000000000000003")
    if NATIVE:
        assert g[3] != np.longdouble(1.0)
    else:
        assert g[3] == 1.0
    if with_string:
        assert sdds.col("s").data[0][0] == "q"
