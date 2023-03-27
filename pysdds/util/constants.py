import numpy as np
import os

# SDDS specification has short, long, float, double, character, or string
# SDDS 2 adds ulong, ushort
# SDDS 4/5 adds long64, ulong64
# Unclear which version adds longdouble - handling this type is very problematic because it changes size based on
# the OS and 32/64 version to ensure byte alignment. Namely, Windows has np.longdouble==np.double, x86 unix = np.float96, x64 unix = np.float128, PowerPC handling is compiler-dependent
_NUMPY_DTYPES = {'short': 'i2', 'ushort': 'u2',
                 'long': 'i4', 'ulong': 'u4',
                 'long64': 'i8', 'ulong64': 'u8',
                 'float': 'f4', 'double': 'f8',
                 'character': object, 'string': object}

# Careful with duplicate keys here
_NUMPY_DTYPES_INV = {np.dtype(np.int16): 'short', np.dtype(np.uint16): 'ushort',
                     np.dtype(np.int32): 'long', np.dtype('<u4'): 'ulong',
                     np.dtype(np.int64): 'long64', np.dtype('<u8'): 'ulong64',
                     np.dtype(np.float64): 'double', np.dtype(np.float32): 'float',
                     object: 'string', np.dtype('O'): 'string'}

# Only add them is likely available, since otherwise np.dtype(np.longdouble) == np.dtype(np.float64)
# which confuses dict type lookups
if os.name == 'posix':
    _NUMPY_DTYPES.update({'longdouble': 'g'})
    _NUMPY_DTYPES_INV.update({np.dtype(np.longdouble): 'longdouble'})

# On all 'reasonable' architectures, things will be little endian, but plenty of old files floating around
_NUMPY_DTYPE_LE = {'short': np.dtype('<i2'), 'ushort': np.dtype('<u2'),
                   'long': np.dtype('<i4'), 'ulong': np.dtype('<u4'),
                   'long64': np.dtype('<i8'), 'ulong64': np.dtype('<u8'),
                   'float': np.dtype('<f4'), 'double': np.dtype('<f8'),
                   'longdouble': np.dtype('<g', align=True),
                   'character': np.dtype('<i1'), 'string': object}

_NUMPY_DTYPE_BE = {'short': np.dtype('>i2'), 'ushort': np.dtype('>u2'),
                   'long': np.dtype('>i4'), 'ulong': np.dtype('>u4'),
                   'long64': np.dtype('>i8'), 'ulong64': np.dtype('>u8'),
                   'float': np.dtype('>f4'), 'double': np.dtype('>f8'),
                   'longdouble': np.dtype('>g'),
                   'character': np.dtype('<i1'), 'string': object}

_NUMPY_DTYPE_FINAL = {'short': np.dtype('i2'), 'ushort': np.dtype('u2'),
                      'long': np.dtype('i4'), 'ulong': np.dtype('u4'),
                      'long64': np.dtype('i8'), 'ulong64': np.dtype('u8'),
                      'float': np.dtype('f4'), 'double': np.dtype('f8'),
                      'longdouble': np.dtype('g'),
                      'character': object, 'string': object}

_PYTHON_TYPE_FINAL = {'short': np.dtype('i2'), 'ushort': np.dtype('u2'),
                      'long': np.dtype('i4'), 'ulong': np.dtype('u4'),
                      'long64': np.dtype('i8'), 'ulong64': np.dtype('u8'),
                      'float': np.dtype('f4'), 'double': np.dtype('f8'),
                      'longdouble': np.dtype('g'),
                      'character': str, 'string': str}

_PYTHON_TYPE_INV = {int: 'long', float: 'double', str: 'string'}
_STRUCT_STRINGS_LE = {'short': '<h', 'ushort': '<H',
                      'long': '<l', 'ulong': '<L',
                      'long64': '<q', 'ulong64': '<Q',
                      'float': '<f', 'double': '<d',
                      'longdouble': '16s',
                      'character': '<c', 'string': object}

_STRUCT_STRINGS_BE = {'short': '>h', 'ushort': '>H',
                      'long': '>l', 'ulong': '>L',
                      'long64': '>q', 'ulong64': '>Q',
                      'float': '>f', 'double': '>d',
                      'longdouble': '16s',
                      'character': '>c', 'string': object}

# Expected field lengths in bytes based on what sddsconvert outputs on x86-64
_NUMPY_DTYPE_SIZES = {'short': 2, 'ushort': 2,
                      'long': 4, 'ulong': 4,
                      'long64': 8, 'ulong64': 8,
                      'float': 4, 'double': 8,
                      'longdouble': 16,
                      'character': 1, 'string': None}

# Expected keys for various SDDS namelists
_KEYS_DESCRIPTION = {'text', 'contents'}
_KEYS_PARAMETER = {'name', 'symbol', 'units', 'description', 'format_string', 'type', 'fixed_value'}
_KEYS_ARRAY = {'name', 'symbol', 'units', 'description', 'format_string', 'type', 'group_name', 'field_length',
               'dimensions'}
_KEYS_COLUMN = {'name', 'symbol', 'units', 'description', 'format_string', 'type', 'field_length'}
_KEYS_DATA = {'mode', 'lines_per_row', 'no_row_counts', 'additional_header_lines', 'column_major_order', 'endian'}
