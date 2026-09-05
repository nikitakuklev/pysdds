import math
from typing import Literal

import numpy as np

# SDDS longdouble fields are 16 bytes: the x87 80-bit extended value in the low 10 bytes (little-endian:
# 64-bit significand with an explicit integer bit, then 15-bit exponent and sign bit), followed by 6 padding bytes.
# numpy exposes this type as np.longdouble only where the C compiler has it (x86-64 gcc/clang: 16 bytes, "float128").
# On Windows (MSVC) and Apple Silicon long double is plain double, so the fields must be decoded by hand.
LONGDOUBLE_FIELD_BYTES = 16
_X87_EXP_BIAS = 16383
_X87_EXP_MAX = 0x7FFF
_X87_INT_BIT = np.uint64(1) << np.uint64(63)


def longdouble_is_native() -> bool:
    """True if np.longdouble is the 16-byte x87 extended type, so SDDS longdouble fields can be viewed directly"""
    return np.dtype(np.longdouble).itemsize == LONGDOUBLE_FIELD_BYTES


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


def decode_longdouble_fields(buffer, endianness: Literal["big", "little"] = "little") -> np.ndarray:
    """Decode packed 16-byte SDDS longdouble fields into a float64 array.

    Used where numpy has no 80-bit type. Values are rounded to the nearest double; anything beyond double range
    becomes inf or 0. Only little-endian fields are supported, since the x87 format itself only exists on
    little-endian hardware.
    """
    if endianness != "little":
        raise NotImplementedError("big-endian longdouble data cannot be decoded without a native 80-bit type")
    raw = np.frombuffer(buffer, dtype=np.uint8)
    if raw.size % LONGDOUBLE_FIELD_BYTES != 0:
        raise ValueError(f"longdouble buffer of {raw.size} bytes is not a multiple of {LONGDOUBLE_FIELD_BYTES}")
    raw = raw.reshape(-1, LONGDOUBLE_FIELD_BYTES)
    mantissa = np.ascontiguousarray(raw[:, :8]).view("<u8").ravel()
    sign_exp = np.ascontiguousarray(raw[:, 8:10]).view("<u2").ravel().astype(np.int64)
    exponent = sign_exp & _X87_EXP_MAX
    negative = (sign_exp & 0x8000) != 0

    with np.errstate(over="ignore", under="ignore"):
        # The significand is an integer scaled by 2^(e - bias - 63); float64() rounds it to 53 bits
        # ldexp takes a C int exponent, which is 32-bit on Windows
        values = np.ldexp(mantissa.astype(np.float64), (exponent - _X87_EXP_BIAS - 63).astype(np.int32))
    # Exponent 0 is zero or denormal (tiny beyond double range either way, ldexp already gave 0);
    # all-ones exponent is inf or nan depending on the fraction bits
    special = exponent == _X87_EXP_MAX
    if special.any():
        fraction = mantissa & ~_X87_INT_BIT
        values[special & (fraction == 0)] = np.inf
        values[special & (fraction != 0)] = np.nan
    values[negative] *= -1
    return values


def encode_longdouble_fields(values, endianness: Literal["big", "little"] = "little") -> bytes:
    """Encode float64 values into packed 16-byte SDDS longdouble fields (x87 extended, zero padded).

    Used where numpy has no 80-bit type. Every double is exactly representable, so this is lossless.
    """
    if endianness != "little":
        raise NotImplementedError("big-endian longdouble data cannot be encoded without a native 80-bit type")
    values = np.ascontiguousarray(values, dtype=np.float64).ravel()
    out = np.zeros((values.size, LONGDOUBLE_FIELD_BYTES), dtype=np.uint8)
    negative = np.signbit(values)
    finite = np.isfinite(values)
    nonzero = values != 0
    normal = finite & nonzero

    mantissa = np.zeros(values.size, dtype=np.uint64)
    exponent = np.zeros(values.size, dtype=np.int64)
    # x = m * 2^e with 0.5 <= |m| < 1, so |m| * 2^64 is an integer in [2^63, 2^64) with the integer bit set
    m, e = np.frexp(np.abs(values[normal]))
    mantissa[normal] = np.ldexp(m, 64).astype(np.uint64)
    exponent[normal] = e + _X87_EXP_BIAS - 1
    # Infinity has the integer bit and a zero fraction, nan has the integer bit and a nonzero fraction
    exponent[~finite] = _X87_EXP_MAX
    mantissa[~finite] = _X87_INT_BIT
    mantissa[np.isnan(values)] |= _X87_INT_BIT >> np.uint64(1)

    out[:, :8] = mantissa.view(np.uint8).reshape(-1, 8)
    sign_exp = (exponent | (negative.astype(np.int64) << 15)).astype("<u2")
    out[:, 8:10] = sign_exp.view(np.uint8).reshape(-1, 2)
    return out.tobytes()
