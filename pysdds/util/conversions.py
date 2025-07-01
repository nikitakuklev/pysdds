import math
from typing import Literal


def float80_to_float64(buffer: bytearray, endianness: Literal["big", "little"]):
    """
    Convert longdouble stored as 16-byte buffer to a standard Python float

    Parameters
    ----------
    buffer
    endianness

    Returns
    -------

    """
    # 80 bit floating point value according to the IEEE-754 specification:
    # 1 bit sign, 15 bit exponent, 1 bit normalization indication, 63 bit mantissa
    # See https://stackoverflow.com/questions/2963055/convert-extended-precision-float-80-bit-to-double-64-bit-in-msvc

    assert len(buffer) == 16
    buffer = buffer[:10]
    if endianness == "little":
        buffer.reverse()

    if (buffer[0] & 0x80) == 0x00:
        sign = 1
    else:
        sign = -1

    exponent = ((buffer[0] & 0x7F) << 8) | buffer[1]

    mantissa = buffer[2:]
    if (mantissa[0] & 0x80) != 0x00:
        normalizeCorrection = 1
    else:
        normalizeCorrection = 0

    m2 = int.from_bytes(mantissa, "big") & 0x7FFFFFFFFFFFFFFF

    value = sign * (normalizeCorrection + float(m2 / (1 << 63))) * math.pow(2, exponent - 16383)
    return value
