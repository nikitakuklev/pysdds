import io

import pysdds


def _read(src: bytes):
    return pysdds.read(io.BytesIO(src))


def test_copy_without_data_with_fixed_value_parameter():
    src = (
        b"SDDS1\n"
        b"&parameter name=f, type=double, fixed_value=1.5, &end\n"
        b"&parameter name=p, type=long, &end\n"
        b"&column name=x, type=double, &end\n"
        b"&data mode=ascii, &end\n"
        b"7\n1\n2.0\n"
    )
    sdds = _read(src)
    empty = sdds.copy(data=False)
    assert empty.n_pages == 0
    assert empty.par("p").data == []
    assert empty.par("f").data == []
    assert empty.col("x").data == []
    # Original untouched
    assert sdds.par("f").data == [1.5]
    assert sdds.par("p").data == [7]


def test_character_fixed_value_parameter():
    src = (
        b"SDDS1\n"
        b"&parameter name=c, type=character, fixed_value=a, &end\n"
        b"&column name=x, type=double, &end\n"
        b"&data mode=ascii, &end\n"
        b"1\n1.0\n"
        b"1\n2.0\n"
    )
    sdds = _read(src)
    assert sdds.n_pages == 2
    assert sdds.par("c").fixed_value == "a"
    assert sdds.par("c").data == ["a", "a"]
    assert sdds.par("c").compare(sdds.copy().par("c"), eps=1e-9)


def test_from_df_numpy_scalar_parameters_and_endianness():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"x": [1.0, 2.0]})
    params = {"f": [np.float64(1.5)], "i": [np.int32(3)], "n": [2**40 + 4], "s": ["abc"]}
    sdds = pysdds.SDDSFile.from_df([df], parameter_dict=params, mode="binary", endianness="big")
    assert [p.type for p in sdds.parameters] == ["double", "long", "long64", "string"]
    assert sdds.endianness == "big"

    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    assert b"!# big-endian" in buf.getvalue()
    sdds2 = pysdds.read(io.BytesIO(buf.getvalue()))
    assert sdds2.par("f").data == [1.5]
    assert sdds2.par("i").data == [3]
    assert sdds2.par("n").data == [2**40 + 4]
    assert sdds2.par("s").data == ["abc"]


def test_write_from_scratch_without_data_namelist():
    """Objects built with add_column/add_parameter and no set_mode() call must still write a readable file"""
    import numpy as np

    for mode in ("binary", "ascii"):
        sdds = pysdds.SDDSFile()  # data namelist is None here
        sdds.mode = mode
        sdds.add_column("x", "double", data=[np.array([1.0, 2.0])])
        sdds.add_parameter("p", "double", data=[1.5])
        sdds.n_pages = 1
        buf = io.BytesIO()
        pysdds.write(sdds, buf)
        assert sdds.data is None  # caller's object untouched
        sdds2 = pysdds.read(io.BytesIO(buf.getvalue()))
        assert sdds2.mode == mode
        assert sdds2.par("p").data == [1.5]
        assert np.array_equal(sdds2.col("x").data[0], [1.0, 2.0])


# Integer parameter lists may contain ordinary Python ints, whose width is checked against the header type.
_INTEGER_PARAMETER_DTYPES = {
    "short": "i2",
    "ushort": "u2",
    "long": "i4",
    "ulong": "u4",
    "long64": "i8",
    "ulong64": "u8",
}


def test_python_integer_parameter_roundtrips():
    import numpy as np

    for mode in ("ascii", "binary"):
        for endian in ("little", "big"):
            for sdds_type, dtype in _INTEGER_PARAMETER_DTYPES.items():
                limits = np.iinfo(dtype)
                values = [int(limits.min), 7, int(limits.max)]
                source = pysdds.SDDSFile()
                source.n_pages = len(values)
                source.add_parameter("p", sdds_type, data=values)
                source.add_parameter("q", "double", data=[1.5] * len(values))
                source.set_mode(mode)
                source.set_endianness(endian)
                source.validate_data()
                output = io.BytesIO()
                pysdds.write(source, output)
                result = pysdds.read(io.BytesIO(output.getvalue()))
                assert result.par("p").data == values, (mode, endian, sdds_type)
                assert all(isinstance(v, np.dtype(dtype).type) for v in result.par("p").data)
                assert result.par("q").data == [1.5] * len(values)
                assert all(type(v) is int for v in source.par("p").data)


def test_python_integer_parameters_reject_overflow_before_writing():
    import numpy as np
    import pytest

    for sdds_type, dtype in _INTEGER_PARAMETER_DTYPES.items():
        limits = np.iinfo(dtype)
        for value in (int(limits.min) - 1, int(limits.max) + 1):
            for mode in ("ascii", "binary"):
                source = pysdds.SDDSFile()
                source.n_pages = 1
                source.add_parameter("p", sdds_type, data=[value])
                source.set_mode(mode)
                output = io.BytesIO()
                with pytest.raises(ValueError, match=f"outside the range of {sdds_type}"):
                    pysdds.write(source, output)
                assert output.getvalue() == b""


def test_integer_parameter_validation_does_not_coerce_other_types():
    import pytest

    for value in (True, False, 1.5, "7"):
        source = pysdds.SDDSFile()
        source.n_pages = 1
        source.add_parameter("p", "long", data=[value])
        with pytest.raises(Exception, match="Parameter type"):
            source.validate_data()
