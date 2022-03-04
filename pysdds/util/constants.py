import numpy as np

# SDDS specification has short, long, float, double, character, or string
_NUMPY_DTYPES = {'short': 'i2', 'ushort': 'u2', 'long': 'i4', 'float': 'f4', 'double': 'f8',
                 'character': object, 'string': object}

# Need to look into i8 support
_NUMPY_DTYPES_INV = {np.dtype(np.int16): 'short', np.dtype(np.int32): 'long', np.dtype(np.int64): 'long',
                     np.dtype(np.float64): 'double', np.dtype(np.float32): 'float',
                     object: 'string', np.dtype('O'): 'string'}

# On all 'reasonable' architectures, things will be little endian, but plenty of old files floating around
_NUMPY_DTYPE_LE = {'short': np.dtype('<i2'), 'ushort': np.dtype('<u2'), 'long': np.dtype('<i4'),
                   'float': np.dtype('<f4'), 'double': np.dtype('<f8'), 'character': np.dtype('<i1'),
                   'string': object}
_NUMPY_DTYPE_BE = {'short': np.dtype('>i2'), 'ushort': np.dtype('>u2'), 'long': np.dtype('>i4'),
                   'float': np.dtype('>f4'), 'double': np.dtype('>f8'), 'character': np.dtype('>i1'),
                   'string': object}
_NUMPY_DTYPE_FINAL = {'short': np.dtype('i2'), 'ushort': np.dtype('u2'), 'long': np.dtype('i4'),
                      'float': np.dtype('f4'), 'double': np.dtype('f8'), 'character': object,
                      'string': object}
_PYTHON_TYPE_FINAL = {'short': np.dtype('i2'), 'ushort': np.dtype('u2'), 'long': np.dtype('i4'),
                      'float': np.dtype('f4'), 'double': np.dtype('f8'), 'character': str,
                      'string': str}
_PYTHON_TYPE_INV = {int: 'long', float: 'double', str: 'string'}
_STRUCT_STRINGS_LE = {'short': '<h', 'ushort': '<H', 'long': '<l',
                      'float': '<f', 'double': '<d', 'character': '<c', 'string': object}
_STRUCT_STRINGS_BE = {'short': '>h', 'ushort': '>H', 'long': '>l',
                      'float': '>f', 'double': '>d', 'character': '>c', 'string': object}

# Expected field lengths in bytes for 32bit architecture (which SDDS was made for initially)
_NUMPY_DTYPE_SIZES = {'short': 2, 'ushort': 2, 'long': 4, 'float': 4, 'double': 8, 'character': 1, 'string': None}

# Expected keys for various SDDS namelists
_KEYS_DESCRIPTION = {'text', 'contents'}
_KEYS_PARAMETER = {'name', 'symbol', 'units', 'description', 'format_string', 'type', 'fixed_value'}
_KEYS_ARRAY = {'name', 'symbol', 'units', 'description', 'format_string', 'type', 'group_name', 'field_length',
               'dimensions'}
_KEYS_COLUMN = {'name', 'symbol', 'units', 'description', 'format_string', 'type', 'field_length'}
_KEYS_DATA = {'mode', 'lines_per_row', 'no_row_counts', 'additional_header_lines', 'column_major_order', 'endian'}
