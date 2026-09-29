"""Declared parameter scalar types are checked before ordinary or streaming output."""

import io
from enum import IntEnum

import numpy as np
import pytest

import pysdds
from pysdds.util.constants import _LONGDOUBLE_NATIVE, _NUMPY_DTYPE_FINAL


class _KeepOpen(io.BytesIO):
    def close(self):
        pass


TRANSPORTS = [(mode, endian) for mode in ("ascii", "binary", "stream") for endian in ("little", "big")]
INTEGER_TYPES = ["short", "ushort", "long", "ulong", "long64", "ulong64"]
FLOAT_TYPES = ["float", "double", "longdouble"]


def _source(sdds_type, values, mode, endian):
    source = pysdds.SDDSFile(add_data_nm=True)
    source.n_pages = 1
    source.set_mode("binary" if mode == "stream" else mode)
    source.set_endianness(endian)
    source.add_parameter("fixed", "double", fixed_value=1.5)
    for i, value in enumerate(values):
        source.add_parameter(f"p{i}", sdds_type, data=[value])
    return source


def _roundtrip(sdds_type, values, mode, endian):
    if sdds_type == "longdouble" and not _LONGDOUBLE_NATIVE and endian == "big" and mode != "ascii":
        pytest.skip("big-endian longdouble writing requires the native extended type")
    source = _source(sdds_type, values, mode, endian)
    output = _KeepOpen()
    if mode == "stream":
        writer = source.copy(data=False).get_streaming_writer(output)
        writer.binary_fixed_rowcount = 0
        writer.begin()
        writer.new_page(values)
        writer.close()
    else:
        pysdds.write(source, output)
    assert all(source.par(f"p{i}").data[0] is value for i, value in enumerate(values))
    result = pysdds.read(io.BytesIO(output.getvalue()), allow_longdouble=True)
    assert result.par("fixed").data == [1.5]
    return [result.par(f"p{i}").data[0] for i in range(len(values))]


@pytest.mark.parametrize("mode,endian", TRANSPORTS)
@pytest.mark.parametrize("sdds_type", INTEGER_TYPES)
def test_integer_scalar_boundaries_and_numpy_width_conversion(mode, endian, sdds_type):
    dtype = _NUMPY_DTYPE_FINAL[sdds_type]
    bounds = np.iinfo(dtype)
    values = [int(bounds.min), int(bounds.max), np.int64(7), np.uint64(9)]
    result = _roundtrip(sdds_type, values, mode, endian)
    assert result == values
    assert all(isinstance(value, dtype.type) for value in result)


@pytest.mark.parametrize("mode,endian", TRANSPORTS)
@pytest.mark.parametrize("sdds_type", FLOAT_TYPES)
def test_float_scalar_conversion_rounding_and_special_values(mode, endian, sdds_type):
    dtype = _NUMPY_DTYPE_FINAL[sdds_type]
    values = [7, np.int64(9), 0.1, np.float32(0.25), -0.0, float("nan"), float("inf"), -float("inf")]
    result = _roundtrip(sdds_type, values, mode, endian)
    for actual, original in zip(result, values):
        expected = dtype.type(original)
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == expected
            assert np.signbit(actual) == np.signbit(expected)


@pytest.mark.parametrize("mode,endian", TRANSPORTS)
@pytest.mark.parametrize("sdds_type", FLOAT_TYPES)
def test_float_finite_range_boundaries(mode, endian, sdds_type):
    dtype = _NUMPY_DTYPE_FINAL[sdds_type]
    limit = np.finfo(dtype).max
    values = [limit, -limit]
    result = _roundtrip(sdds_type, values, mode, endian)
    assert all(np.isfinite(value) for value in result)
    assert result == values


class _IntegerEnum(IntEnum):
    VALUE = 7


INVALID_VALUES = [
    pytest.param("long", 2**31, ValueError, id="python-long-overflow"),
    pytest.param("long", np.int64(2**31), ValueError, id="numpy-long-overflow"),
    pytest.param("ulong", np.int64(-1), ValueError, id="numpy-negative-unsigned"),
    pytest.param("ushort", np.uint64(2**16), ValueError, id="numpy-ushort-overflow"),
    pytest.param("long64", np.uint64(2**63), ValueError, id="numpy-long64-overflow"),
    pytest.param("ulong64", -1, ValueError, id="python-negative-unsigned"),
    pytest.param("float", 1e39, ValueError, id="python-float-overflow"),
    pytest.param("float", np.float64(1e39), ValueError, id="numpy-float-overflow"),
    pytest.param("float", int(np.finfo(np.float32).max) + 1, ValueError, id="integer-float-overflow"),
    pytest.param("double", 2**1024, ValueError, id="integer-double-overflow"),
    pytest.param("longdouble", int(np.finfo(np.longdouble).max) + 1, ValueError, id="integer-longdouble-overflow"),
    pytest.param("long", 1.5, TypeError, id="fractional-integer"),
    pytest.param("long", np.float64(1.5), TypeError, id="numpy-fractional-integer"),
    pytest.param("long", _IntegerEnum.VALUE, TypeError, id="enum-integer"),
    pytest.param("double", _IntegerEnum.VALUE, TypeError, id="enum-double"),
    pytest.param("long", True, TypeError, id="bool-integer"),
    pytest.param("double", True, TypeError, id="bool-double"),
    pytest.param("double", np.bool_(True), TypeError, id="numpy-bool-double"),
    pytest.param("double", "7", TypeError, id="string-double"),
    pytest.param("double", 7 + 0j, TypeError, id="complex-double"),
    pytest.param("string", 7, TypeError, id="integer-string"),
    pytest.param("character", "ab", ValueError, id="wide-character"),
    pytest.param("character", "", ValueError, id="empty-character"),
    pytest.param("character", "\u0100", ValueError, id="character-outside-byte-range"),
    pytest.param("string", "\N{GREEK SMALL LETTER ALPHA}", ValueError, id="nonascii-string"),
]


@pytest.mark.parametrize("mode,endian", TRANSPORTS)
@pytest.mark.parametrize("sdds_type,value,error", INVALID_VALUES)
def test_invalid_later_parameter_leaves_output_and_state_unchanged(mode, endian, sdds_type, value, error):
    source = _source("long", [7], mode, endian)
    source.add_parameter("bad", sdds_type, data=[value])
    output = _KeepOpen()
    original_values = [7, value]
    if mode == "stream":
        if sdds_type == "longdouble" and not _LONGDOUBLE_NATIVE and endian == "big":
            pytest.skip("big-endian longdouble writing requires the native extended type")
        writer = source.copy(data=False).get_streaming_writer(output)
        writer.binary_fixed_rowcount = 0
        writer.begin()
        before = output.getvalue(), output.tell(), writer.write_stage, writer.current_page
        with pytest.raises(error):
            writer.new_page(original_values)
        assert (output.getvalue(), output.tell(), writer.write_stage, writer.current_page) == before
        valid = "x" if sdds_type in ("string", "character") else 3
        writer.new_page([7, valid])
        writer.close()
        result = pysdds.read(io.BytesIO(output.getvalue()), allow_longdouble=True)
        assert result.par("fixed").data == [1.5]
        assert result.par("p0").data == [7]
        assert result.par("bad").data == [valid]
    else:
        output.write(b"existing bytes")
        before = output.getvalue(), output.tell()
        with pytest.raises(error):
            pysdds.write(source, output)
        assert (output.getvalue(), output.tell()) == before
    assert original_values[1] is value
    assert source.par("bad").data[0] is value


@pytest.mark.skipif(not _LONGDOUBLE_NATIVE, reason="requires native extended-precision SDDS longdouble")
@pytest.mark.parametrize("mode,endian", TRANSPORTS)
def test_longdouble_preserves_precision_and_large_integer_range(mode, endian):
    precise = np.longdouble(1) + np.finfo(np.longdouble).eps
    enormous = 2**16000
    values = [precise, np.longdouble("1e400"), enormous, -enormous]
    result = _roundtrip("longdouble", values, mode, endian)
    assert result[0] == precise and result[0] != np.longdouble(1)
    assert result[1] == values[1]
    assert result[2] == np.ldexp(np.longdouble(1), 16000)
    assert result[3] == -result[2]


@pytest.mark.skipif(not _LONGDOUBLE_NATIVE, reason="requires a finite value beyond binary64 range")
@pytest.mark.parametrize("mode,endian", TRANSPORTS)
def test_finite_longdouble_cannot_overflow_double(mode, endian):
    source = _source("double", [np.longdouble("1e400")], mode, endian)
    output = _KeepOpen()
    if mode == "stream":
        writer = source.copy(data=False).get_streaming_writer(output)
        writer.begin()
        before = output.getvalue()
        with pytest.raises(ValueError, match="outside the range of double"):
            writer.new_page([np.longdouble("1e400")])
        assert output.getvalue() == before
        writer.close()
    else:
        with pytest.raises(ValueError, match="outside the range of double"):
            pysdds.write(source, output)
        assert output.getvalue() == b""


@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize("codepoint", [128, 255])
def test_ascii_character_parameter_retains_single_byte_range(endian, codepoint):
    value = chr(codepoint)
    assert _roundtrip("character", [value], "ascii", endian) == [value]


@pytest.mark.parametrize("mode", ["binary", "stream"])
@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize("codepoint", [128, 255])
def test_binary_character_parameter_writes_unsigned_byte(mode, endian, codepoint):
    source = pysdds.SDDSFile(add_data_nm=True)
    source.n_pages = 1
    source.set_mode("binary")
    source.set_endianness(endian)
    source.add_parameter("c", "character", data=[chr(codepoint)])
    output = _KeepOpen()
    if mode == "stream":
        writer = source.copy(data=False).get_streaming_writer(output)
        writer.binary_fixed_rowcount = 0
        writer.begin()
        writer.new_page([chr(codepoint)])
        writer.close()
    else:
        pysdds.write(source, output)
    payload = output.getvalue().split(b"&data", 1)[1].split(b"\n", 1)[1]
    assert payload == bytes(4) + bytes([codepoint])
