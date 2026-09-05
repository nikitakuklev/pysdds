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
    params = {"f": [np.float64(1.5)], "i": [np.int32(3)], "n": [4], "s": ["abc"]}
    sdds = pysdds.SDDSFile.from_df([df], parameter_dict=params, mode="binary", endianness="big")
    assert [p.type for p in sdds.parameters] == ["double", "long", "long", "string"]
    assert sdds.endianness == "big"

    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    assert b"!# big-endian" in buf.getvalue()
    sdds2 = pysdds.read(io.BytesIO(buf.getvalue()))
    assert sdds2.par("f").data == [1.5]
    assert sdds2.par("i").data == [3]
    assert sdds2.par("n").data == [4]
    assert sdds2.par("s").data == ["abc"]
