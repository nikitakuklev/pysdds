import io
import logging
import time
import shlex
from pathlib import Path
from typing import Union, Iterable, List, IO, BinaryIO, Optional
import struct

import numpy as np
import pandas as pd

from ..structures import *

# The proper way to implement conditional logging is to check current level,
# but this creates too much overhead in hot loops. So, old school global vars it is.
logger = logging.getLogger(__name__)
DEBUG2 = False  # one more level from debug
TRACE = False  # two more levels from debug

# SDDS specification has short, long, float, double, character, or string
_NUMPY_DTYPES = {'short': 'i2', 'ushort': 'u2', 'long': 'i4', 'float': 'f4', 'double': 'f8',
                 'character': object, 'string': object}

# On all 'reasonable' architectures, things will be little endian, but plenty of old files floating around
_NUMPY_DTYPE_STRINGS_LE = {'short': np.dtype('<i2'), 'ushort': np.dtype('<u2'), 'long': np.dtype('<i4'),
                           'float': np.dtype('<f4'), 'double': np.dtype('<f8'), 'character': np.dtype('<i1'),
                           'string': object}
_NUMPY_DTYPE_STRINGS_BE = {'short': np.dtype('>i2'), 'ushort': np.dtype('>u2'), 'long': np.dtype('>i4'),
                           'float': np.dtype('>f4'), 'double': np.dtype('>f8'), 'character': np.dtype('>i1'),
                           'string': object}
_NUMPY_DTYPE_FINAL = {'short': np.dtype('i2'), 'ushort': np.dtype('u2'), 'long': np.dtype('i4'),
                      'float': np.dtype('f4'), 'double': np.dtype('f8'), 'character': object,
                      'string': object}
_STRUCT_DTYPE_STRINGS_LE = {'short': '<h', 'ushort': '<H', 'long': '<l',
                            'float': '<f', 'double': '<d', 'character': '<c', 'string': object}
_STRUCT_DTYPE_STRINGS_BE = {'short': '>h', 'ushort': '>H', 'long': '>l',
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

_ASCII_TEXT_PARSE_METHOD = 'read_table'
# _ASCII_TEXT_PARSE_METHOD = 'shlex'
_ASCII_NUMERIC_PARSE_METHOD = 'read_table'  # 'fromtxt'


def _open_file(filepath: Path, compression: str, use_magic_values: bool = False) -> IO[bytes]:
    """Open the file path for reading as a raw byte stream. Compression is determined based on either file extension,
    or magic strings If no matches are found, file is assumed to be uncompressed

    Parameters
    ----------
    filepath : Path
        Path object
    compression : str
        Compression mode
    use_magic_values : bool
        If True, and `compression` == 'auto', then magic byte signatures will be used for format determination.
        If false, file extension is used.

    Returns
    -------
    stream : BufferedIOReader
        The opened, buffered file stream

    """
    assert isinstance(filepath, Path)

    if not filepath.is_file():
        raise IOError(f'File ({filepath}) does not exist or cannot be read')

    # Check file size
    filesize = filepath.stat().st_size
    if filesize > 1e9:
        logger.warning(f'File size ({filesize / 1e9:.2f})MB is quite large and will cause performance issues')

    if compression is not None and compression not in ['auto', 'xz', 'gz', 'bz2']:
        raise ValueError(f'Compression format ({compression}) is not recognized')

    if compression == 'auto':
        if use_magic_values:
            # https://www.garykessler.net/library/file_sigs.html
            magic_dict = {
                b"\xfd\x37\x7A\x58\x5A\x00": 'xz',
                b"\x1f\x8b\x08": 'gz',
                b"\x42\x5a\x68": 'bz2',
                b"\x50\x4b\x03\x04": 'zip',
            }
            max_len = max(len(x) for x in magic_dict.keys())
            with open(filepath, 'rb') as f:
                magic_bytes = f.read(max_len)
                match = None
                for magic, c in magic_dict.items():
                    if magic_bytes.startswith(magic):
                        match = c
                        break
                compression = match  # None->no compression
            logger.info(f'Auto compression resolved as ({compression}) from file byte signature')
        else:
            extension = filepath.suffix.strip('.')
            if extension in ['xz', '7z', 'lzma']:
                compression = 'xz'
            elif extension in ['gz', 'bz2', 'zip']:
                compression = extension
            else:
                compression = None
            logger.info(f'Auto compression resolved as ({compression}) from file extension')

    # By default, Python reads in small blocks of io.DEFAULT_BUFFER_SIZE = 8192
    # To encourage large reads and fully consuming small files, a large 2MB buffer is specified
    # For decompression, the decompress function is called on small blocks regardless, so we supply
    # a buffered reader to at least read large chunks from network storage
    # See bpo-41486 for Python 3.10 patch that will improve decompress performance
    try:
        buffered_stream = open(filepath, 'rb', buffering=2097152)  # 2**20
        if compression == 'xz':
            import lzma
            stream = lzma.open(buffered_stream, 'rb')
        elif compression == 'gz':
            import gzip
            stream = gzip.open(buffered_stream, 'rb')
        elif compression == 'bz2':
            import bz2
            stream = bz2.open(buffered_stream, 'rb')
        elif compression == 'zip':
            import zipfile
            stream = zipfile.ZipFile(buffered_stream, 'r')
        else:
            stream = buffered_stream
        if TRACE:
            logger.debug(f'File stream: {buffered_stream}')
            logger.debug(f'Final stream: {stream}')
        return stream
    except IOError as ex:
        logger.exception(f'File {str(filepath)} IO failed')
        raise ex


def read(filepath: Union[Path, str],
         mode: Optional[str] = 'auto',
         endianness: Optional[str] = 'auto',
         compression: Optional[str] = 'auto',
         pages: Optional[Iterable[int]] = None,
         arrays: Optional[Union[Iterable[int], Iterable[str]]] = None,
         cols: Optional[Union[Iterable[int], Iterable[str]]] = None,
         header_only: Optional[bool] = False) -> SDDSFile:
    """Read in an SDDS file

    Parameters
    ----------
    filepath : str or Path
        A valid absolute or relative file path, or a concrete Path object
    mode : str, optional
        SDDS mode to use for reading the file, one of 'auto', 'binary', or 'ascii'. Defaults to 'auto'.
    endianness : str, optional
        Endianness to use for numerical values, one of 'auto', 'big', 'little'. Defaults to 'auto'.
    compression : str, optional
        Compression format to assume for the stream, one of None, 'auto', 'xz', 'gz', 'bz2', 'zip'. Defaults to 'auto'.
        If file is compressed, performance for large files on Python <= 3.9 will be slow due a buffering regression.
    header_only : bool, optional
        If True, only the header is parsed. This leaves SDDS object in an invalid state with no data structures,
        but increases performance when only header metadata is of interest.
    pages : array-like, optional
        If given, the list of page indices to read (starting from page 0). Note that page numbers in resulting SDDSFile
        will be renumbered in sequence. In other words, if pages=[0,2], valid indices for SDDSFile data would be page=0
        and page=1. This behaviour might change in the future.
    arrays : array-like, optional
        If given, the list of arrays to read. All entries must be either array indices or names.
    cols : array-like, optional
        If given, the list of columns to read. All entries must be either column indices or names.


    Returns
    -------
    sdds_file: SDDSFile
        Instance containing the read data
    """

    # Argument verification
    if isinstance(filepath, str):
        filepath = Path(filepath)
    elif isinstance(filepath, Path):
        pass
    else:
        raise Exception('Filepath is not a string or Path object')

    # Array consistency
    array_mask_mode = 0
    if arrays is not None:
        if all(isinstance(el, str) for el in arrays):
            array_mask_mode = 1
        elif all(isinstance(el, int) for el in arrays):
            array_mask_mode = 2
        else:
            raise ValueError(f'Array selection ({arrays}) is neither all strings nor all integer indices')

    # Column consistency
    column_mask_mode = 0
    if cols is not None:
        if all(isinstance(c, str) for c in cols):
            column_mask_mode = 1
        elif all(isinstance(c, int) for c in cols):
            column_mask_mode = 2
        else:
            raise ValueError(f'Column selection ({cols}) is neither all strings nor all integer indices')

    if mode not in ['auto', 'binary', 'ascii']:
        raise ValueError(f'SDDS mode ({mode}) is not recognized')

    if endianness not in ['auto', 'big', 'little']:
        raise ValueError(f'SDDS binary endianness ({endianness}) is not recognized')

    if compression not in [None, 'auto', 'xz', 'gz', 'bz2', 'zip']:
        raise ValueError(f'SDDS compression ({compression}) is not recognized')

    if pages is not None:
        try:
            assert all(isinstance(i, int) for i in pages)
            # We will not know how many pages are present until reading file, so create mask now
            pages_mask = [i in pages for i in range(0, max(pages) + 1)]
            pages = np.array(pages)
        except Exception:
            raise ValueError(f'Pagelist is not an array-like object of ints')
    else:
        pages_mask = None
        pages = None

    sdds = SDDSFile()
    sdds.__source_file = str(filepath)

    logger.info(f'Opening file "%s"', str(filepath))
    logger.info(f'Mode (%s), compression (%s), endianness (%s)', mode, compression, endianness)
    t_start = time.perf_counter()
    # File is opened in binary mode because it is necessary for data parsing,
    # reopening after header parsing would interfere with IO buffering
    file = _open_file(filepath, compression, use_magic_values=False)
    sdds._source_file_size = filepath.stat().st_size
    try:
        # First, read the header
        _read_header_fullstream(file, sdds, mode, endianness)
        logger.info(f'Header parsed: {len(sdds.parameters)} parameters, {len(sdds.arrays)} arrays,'
                    f' {len(sdds.columns)} columns')
        if TRACE:
            logger.debug(f'Params: {sdds.parameters}')
            logger.debug(f'Arrays: {sdds.arrays}')
            logger.debug(f'Columns: {sdds.columns}')

        if header_only:
            return sdds

        if sdds._meta_fixed_rowcount:
            if sdds.mode != 'binary':
                raise ValueError(f'Meta-command "!#fixed-rowcount" requires binary mode, not {sdds.mode}')

        # Verify that parameter, array, column, and page masks can be applied
        if array_mask_mode == 0:
            array_mask = [True for _ in range(len(sdds.arrays))]
        elif array_mask_mode == 1:
            available_set = set(arrays)
            parsed_names = sdds.array_names
            parsed_set = set(parsed_names)
            items_dict = sdds.array_dict
            if not available_set.issubset(parsed_set):
                raise ValueError(f'Requested arrays ({arrays}) are not a subset of available ({parsed_names})')
            array_mask = [True if el.name in available_set else False for el in sdds.arrays]
            for c in parsed_set.difference(available_set):
                items_dict[c]._enabled = False
        elif array_mask_mode == 2:
            array_mask = [False for _ in range(len(sdds.arrays))]
            for idx in arrays:
                array_mask[idx] = True
        else:
            raise ValueError

        if column_mask_mode == 0:
            column_mask = [True for _ in range(len(sdds.columns))]
        elif column_mask_mode == 1:
            parsed_names = sdds.column_names
            available_set = set(cols)
            parsed_set = set(parsed_names)
            if not available_set.issubset(parsed_set):
                raise ValueError(f'Requested columns ({cols}) are not a subset file data ({parsed_names})')
            column_mask = [True if c.name in available_set else False for c in sdds.columns]
            for c in parsed_set.difference(available_set):
                sdds.columns_dict[c]._enabled = False
        elif column_mask_mode == 2:
            column_mask = [False for _ in range(len(sdds.columns))]
            for column_index in cols:
                column_mask[column_index] = True
        else:
            raise ValueError

        logger.debug('Masks are valid for current header')
        logger.debug(f'Array mask: {array_mask}')
        logger.debug(f'Column mask: {column_mask}')
        logger.debug(f'Page mask: {pages_mask} (actual page count TBD)')

        is_columns_numeric = not any(el.type == 'string' for el in sdds.columns)
        logger.debug(f'Columns numeric: {is_columns_numeric}')

        # Skip lines if necessary
        if sdds.data.additional_header_lines != 0:
            if sdds.mode == 'binary':
                logger.warning('Option "additional_header_lines" will be ignored in binary mode')
            else:
                for i in range(sdds.data.additional_header_lines):
                    file.readline()

        if sdds.mode == 'binary':
            _read_pages_binary(file, sdds, arrays_mask=array_mask, columns_mask=column_mask, pages_mask=pages_mask)
        else:
            # Streaming ascii data is not yet supported because performance is bad
            if sdds.data.lines_per_row != 1:
                raise NotImplementedError("lines_per_row != 1 is not yet supported")

            if sdds.data.no_row_counts != 0:
                raise NotImplementedError("no_row_counts != 0 is not yet supported")

            if is_columns_numeric:
                logger.debug(f'Calling ASCII numeric column parser')
                _read_pages_ascii_numeric_lines(file, sdds, arrays_mask=array_mask, columns_mask=column_mask,
                                                pages_mask=pages_mask)
            else:
                logger.debug(f'Calling ASCII mixed column parser with method {_ASCII_TEXT_PARSE_METHOD}')
                _read_pages_ascii_mixed_lines(file, sdds, arrays_mask=array_mask, columns_mask=column_mask,
                                              pages_mask=pages_mask)

        if pages is not None:
            if sdds.n_pages != len(pages):
                raise IOError(f'Parser failed - got {sdds.n_pages} pages while {len(pages)} were requested')
    finally:
        file.close()

    cols_enabled = [c for c in sdds.columns if c._enabled]
    n_cols_enabled = len(cols_enabled)
    arrays_enabled = sum(1 for c in sdds.arrays if c._enabled)
    if len(cols_enabled) > 0:
        n_rows = sum(len(v) for v in cols_enabled[0].data)
    else:
        n_rows = 0
    logger.info(f'Finished in {(time.perf_counter() - t_start) * 1e3:.3f} ms')
    logger.info(f'Totals: {sdds.n_pages} pages, {n_rows} rows, {len(sdds.parameters)} parameters,'
                f' {arrays_enabled}/{len(sdds.arrays)} arrays, {n_cols_enabled}/{len(sdds.columns)} columns\n')
    logger.debug(f'File description:')
    logger.debug(f'{sdds.describe()}')
    return sdds


def __get_next_line(stream: IO[bytes], accept_meta_commands: bool = True, strip: bool = False) -> str:
    """ Find next line that has valid SDDS data """
    while True:
        line = stream.readline().decode('ascii')
        if len(line) == 0:
            # EOF
            return None

        if '!' in line:
            # Even though both 'in' and 'find' operations are aliased to C code, using 'in' as initial check is
            # significantly faster (~10x), and makes sense since midpoint comments are expected to be rare
            if line.startswith('!'):
                # Full comment line, most common case
                if line.startswith('!#'):
                    # Meta-command
                    if accept_meta_commands:
                        return line
                    else:
                        raise ValueError(f'Meta-command {line} encountered unexpectedly')
                else:
                    if TRACE:
                        logger.debug(f'>>NXL | pos %s | SKIP FULL %s', stream.tell(), repr(line))
                    continue
            else:
                # Partial comment line
                idx = line.find('!')
                if TRACE:
                    logger.debug(f'>>NXL | pos {stream.tell()} | SKIP PARTIAL {repr(line)} | ret {line[:idx]}')
                if strip:
                    return line[:idx].strip()
                else:
                    return line[:idx]
        else:
            if strip:
                return line.strip()
            else:
                return line


def _read_header_fullstream(file: IO[bytes], sdds: SDDSFile, mode: str, endianness: str) -> None:
    """
    Read SDDS header - the ASCII text that described the data contained in the SDDS file. This parser uses the common
    approach of ingesting a stream of bytes and checking tokens, meaning it works on more input types but is slower
    than using native python split/find/etc. functions
    """
    logger.debug('Parsing header with streaming reader')
    # Hopefully no malicious files are fed, 10000 length should be enough for normal files
    version_line = file.readline(10000).decode('ascii').rstrip()
    line_num = 1
    if version_line[:4] != 'SDDS' or len(version_line) != 5:
        raise AttributeError(f'Header parsing failed on line {line_num}: {repr(version_line)} is not a valid version')

    try:
        sdds_version = int(version_line[4])
    except Exception:
        raise AttributeError(f'Unrecognized SDDS version: {version_line[5]}')

    if sdds_version > 3:
        raise ValueError(f'This package only supports SDDS version 3 or lower, file is version {sdds_version}')

    logger.debug(f'File version: {version_line}')

    namelists = []

    def __find_next_namelist(stream, accept_meta_commands=False):
        # accumulate multi-line parameters
        line = buffer = __get_next_line(stream, accept_meta_commands)  # file.readline().decode('ascii')
        if accept_meta_commands and line.startswith('!#'):
            return buffer.strip()
        else:
            while not line.rstrip().endswith('&end'):
                line = stream.readline().decode('ascii')
                if line.strip().startswith('!'):
                    logger.debug(f'MULTILINE PAR > COMMENT SKIP {line}')
                    continue
                logger.debug(f'MULTILINE PAR > adding line {line}')
                if len(line) is None:
                    raise Exception('Unexpected EOF')
                buffer += line
            buffer2 = buffer.strip()
        return buffer2

    while True:
        line = __find_next_namelist(file, accept_meta_commands=True)
        # line = file.readline().decode('ascii').strip('\n')
        len_line = len(line)
        line_num += 1
        if TRACE:
            logger.debug(f'Line %d: %s', line_num, line)
        if line.startswith('!'):
            if not line.startswith('!#'):
                raise Exception(line)
            if line == '!# big-endian':
                if endianness == 'auto' or endianness == 'big':
                    sdds.endianness = 'big'
                    logger.debug(f'Binary file endianness set to ({sdds.endianness})')
                else:
                    raise ValueError(f'File endianness ({line}) does not match requested one ({endianness})')
            elif line == '!# little-endian':
                if endianness == 'auto' or endianness == 'little':
                    sdds.endianness = 'little'
                    logger.debug(f'Binary file endianness set to ({sdds.endianness})')
                else:
                    raise ValueError(f'File endianness ({line}) does not match requested one ({endianness})')
            elif line == '!# fixed-rowcount':
                sdds._meta_fixed_rowcount = True
            else:
                raise Exception(f'Meta command {line} is not recognized')
            continue

        assert line[0] == '&'
        pos_end = line.find(' ')
        if pos_end == -1:
            raise Exception
        command = line[0:pos_end]

        # l.debug(f'>Command parse result: {command}')

        def __parse_namelist_entry():
            nonlocal line, pos
            tokens = []
            no_chars_allowed = False
            while True:
                #logger.debug(f'Char {line[pos]} | tokens {tokens} | nca {no_chars_allowed}')
                if line[pos] == ' ':
                    no_chars_allowed = True
                elif line[pos] == '=':
                    # Transition to value
                    key = ''.join(tokens)
                    break
                else:
                    if no_chars_allowed:
                        raise Exception
                    else:
                        tokens.append(line[pos])
                pos += 1
                if pos >= len_line:
                    raise Exception(f'End of line reached')
            if key == '':
                raise Exception

            pos += 1
            no_chars_allowed = False
            literal_mode = False
            value_tokens = []
            while True:
                c = line[pos]
                #logger.debug(f'{pos} | char {c} | nca {no_chars_allowed} | l {literal_mode} | tokens {value_tokens}')
                if literal_mode:
                    # Inside the quotes
                    if c == '"':
                        # Toggle literal mode
                        literal_mode = False
                    else:
                        # Append anything else
                        value_tokens.append(c)
                else:
                    # Outside the quotes
                    if c == ' ':
                        # At end of namelist
                        no_chars_allowed = True
                    elif c == '"':
                        # Toggle literal mode
                        literal_mode = True
                    elif c == ',':
                        value = ''.join(value_tokens)
                        pos += 1
                        break
                    else:
                        if no_chars_allowed:
                            raise Exception
                        else:
                            value_tokens.append(c)
                pos += 1
                if pos >= len_line:
                    raise Exception(
                        f'End of line reached | {pos} | char {c} | tokens {value_tokens} | nca {no_chars_allowed}')

            if literal_mode:
                raise Exception

            # l.debug(f'>>NM key: {key}')
            # l.debug(f'>>NM value: {value}')
            return key, value

        # Main nameloop iteration
        nm_dict = {}
        pos = pos_end
        while pos < len(line):
            if line[pos] == ' ':
                # skip
                pos += 1
            elif line[pos] == '&':
                if line[pos + 1:pos + 4] == 'end':
                    # end of namelist
                    break
                else:
                    # unrecognized command end
                    raise Exception
            else:
                k, v = __parse_namelist_entry()
                nm_dict[k] = v

        logger.debug(f'>Parse result %s | %s', command, nm_dict)
        # hack around newlines in escaped strings
        if command == '&parameter':
            if 'fixed_value' in nm_dict:
                nm_dict['fixed_value'] = nm_dict['fixed_value'].replace('\n', ' ')

        nm_keys = set(nm_dict.keys())
        namelists.append(nm_dict)

        # expected_keys = None
        if command == '&description':
            if sdds.description is not None:
                raise ValueError('Duplicate description entry found')
            expected_keys = _KEYS_DESCRIPTION
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f'Namelist keys {nm_keys} unexpected for namelist {command}')
            sdds.description = Description(nm_dict)
        elif command == '&parameter':
            expected_keys = _KEYS_PARAMETER
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f'Namelist keys {nm_keys} unexpected for namelist {command}')
            sdds.parameters.append(Parameter(nm_dict, sdds=sdds))
        elif command == '&array':
            expected_keys = _KEYS_ARRAY
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f'Namelist keys {nm_keys} unexpected for namelist {command}')
            sdds.arrays.append(Array(nm_dict, sdds=sdds))
        elif command == '&column':
            expected_keys = _KEYS_COLUMN
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f'Namelist keys {nm_keys} unexpected for namelist {command}')
            sdds.columns.append(Column(nm_dict))
        elif command == '&data':
            # This should be last command
            expected_keys = _KEYS_DATA
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f'Namelist keys {nm_keys} unexpected for namelist {command}')
            file_mode = nm_dict['mode']
            if mode != 'auto':
                assert mode == file_mode
            else:
                if not (file_mode == 'binary' or file_mode == 'ascii'):
                    raise Exception(f'Unrecognized mode ({file_mode}) found in file')
            sdds.mode = file_mode
            sdds.data = Data(nm_dict)
            break
        elif command == '&include':
            raise ValueError('This package does not support &include namelist command')
        else:
            raise ValueError(f'Unrecognized namelist command {command} on line {line_num}')

    if sdds.columns or sdds.parameters or sdds.arrays:
        if sdds.data is None:
            raise AttributeError('SDDS file contains columns, arrays, or parameters - &data namelist is required')

    sdds.n_parameters = len(sdds.parameters)
    sdds.n_arrays = len(sdds.arrays)
    sdds.n_columns = len(sdds.columns)


def _read_pages_binary(file: IO[bytes],
                       sdds: SDDSFile,
                       arrays_mask: List[bool],
                       columns_mask: List[bool],
                       pages_mask: Optional[List[bool]],
                       convert_to_native_endianness: bool = True):
    """Read a binary page of SDDS file. Page length is given by 4 bytes ('long') at the start of the segment.

    If file contains string fields, we will have to ingest input stream to determine the length of each string (given
    by a 'long' preceding it). This slows down parsing significantly, and prevents efficient seeking of next page.

    If there are no strings, data length is fixed and full page buffer can be read directly into numpy.

    Parameter section of the file is always parsed in 'slow string' mode, but column section will use fast path if
    possible (independent of parameter string contents).

    In order to be more pythonic, this method should be written using separate generator functions,
    but for performance reasons manual inlining and other tweaks had to be done.

    Parameters
    ----------
    file: IObuffer
    sdds: the SDDSFile to store data in
    columns_mask: columns bitmask

    Returns
    -------
    None
    """
    # Set up endianness
    endianness = sdds.endianness
    flip_bytes = False
    if endianness == 'big':
        NUMPY_DTYPE_STRINGS = _NUMPY_DTYPE_STRINGS_BE
        STRUCT_DTYPE_STRINGS = _STRUCT_DTYPE_STRINGS_BE
        if convert_to_native_endianness:
            flip_bytes = True
    elif endianness == 'little':
        NUMPY_DTYPE_STRINGS = _NUMPY_DTYPE_STRINGS_LE
        STRUCT_DTYPE_STRINGS = _STRUCT_DTYPE_STRINGS_LE
    else:
        raise ValueError(f'SDDS endianness ({endianness}) is invalid')
    length_dtype = np.dtype(NUMPY_DTYPE_STRINGS['long'])

    # Assemble data types - parameters are first on every page, but those with fixed_value are skipped
    parameters = []
    parameter_types = []
    parameter_lengths: List[Optional[int]] = []
    for p in sdds.parameters:
        if not p.fixed_value:
            parameters.append(p)
            t = p.type
            if t == 'string':
                parameter_types.append(None)
                parameter_lengths.append(None)
            else:
                numpy_type = NUMPY_DTYPE_STRINGS[t]
                parameter_types.append(numpy_type)
                parameter_lengths.append(_NUMPY_DTYPE_SIZES[t])
    logger.debug(f'Parameters to parse: {len(parameters)} of {len(sdds.parameters)}')
    logger.debug(f'Parameter types: {parameter_types}')
    logger.debug(f'Parameter lengths: {parameter_lengths}')
    n_parameters = len(parameters)

    # Arrays go here
    arrays = sdds.arrays
    arrays_type = []
    arrays_size: List[Optional[int]] = []
    for i, a in enumerate(arrays):
        t = a.type
        if t == 'string':
            mapped_t = object
        else:
            mapped_t = NUMPY_DTYPE_STRINGS[t]
        arrays_type.append(mapped_t)
        arrays_size.append(_NUMPY_DTYPE_SIZES[t])
    n_arrays = len(arrays_type)
    logger.debug(f'Arrays to parse: {len(arrays_type)}')
    logger.debug(f'Array types: {arrays_type}')
    logger.debug(f'Array lengths: {arrays_size}')

    # Columns follow arrays
    columns_type = []
    columns_type_struct = []
    columns_structs = []
    columns_store_type = []
    columns_len: List[Optional[int]] = []
    combined_struct = combined_size = combined_dtype = None
    columns = sdds.columns
    for i, c in enumerate(columns):
        t = c.type
        columns_type.append(NUMPY_DTYPE_STRINGS[t])
        columns_store_type.append(_NUMPY_DTYPE_FINAL[t])
        columns_len.append(_NUMPY_DTYPE_SIZES[t])
        columns_type_struct.append(STRUCT_DTYPE_STRINGS[t])
        columns_structs.append(struct.Struct(STRUCT_DTYPE_STRINGS[t]) if t != 'string' else None)
    n_columns = len(columns_type)
    columns_all_numeric = object not in columns_type
    if columns_all_numeric:
        combined_struct = columns_type_struct[0]
        for v in columns_type_struct[1:]:
            combined_struct += v[1]
        combined_size = sum(columns_len)
    logger.debug(f'Columns to parse: {len(columns_type)}')
    logger.debug(f'Column types: {columns_type}')
    logger.debug(f'Column lengths: {columns_len}')
    logger.debug(f'All numeric: {columns_all_numeric}')

    if sdds.data.column_major_order != 0:
        pass
    elif columns_all_numeric and sdds._meta_fixed_rowcount:
        # Numeric types but fixed rows - have to parse row by row
        logger.debug('All columns are numeric and data is row order -> reading whole rows')
    elif columns_all_numeric and not sdds._meta_fixed_rowcount and sdds._source_file_size > 500e6:
        # Row by row parsing with single struct - memory efficient and fast
        logger.debug('All columns numeric, no fixed rows, data is row order, large size -> using row-wide struct')
    elif columns_all_numeric and not sdds._meta_fixed_rowcount:
        # Whole page parsing by using buffer as structured array - the fastest, zero copy method
        logger.debug(f'All columns numeric, no fixed rows, row order, small size -> using structured array')
        # must specify endianness, or linux/windows struct lengths will differ!!!
        combined_dtype = np.dtype([(str(i), c.descr[0][1]) for i, c in enumerate(columns_type)])
    else:
        # Most general row-order parser
        logger.debug('Not all columns numeric, data is row order -> using slow sequential parser')

    page_idx = 0
    page_stored_idx = 0
    while True:
        columns_data = []

        # Main loop
        if pages_mask is not None:
            if page_idx >= len(pages_mask):
                raise Exception('Should not be reachable')
            else:
                page_skip = not pages_mask[page_idx]
        else:
            page_skip = False

        # Read page size
        byte_array = file.read(4)
        assert len(byte_array) == 4
        page_size = int.from_bytes(byte_array, endianness)
        if not 0 <= page_size <= 1e7:
            raise ValueError(
                f'Page size ({page_size}) ({byte_array}) is unreasonable - is file not {endianness}-endian?')
        logger.debug(f'Page {page_idx} size is {page_size} | {byte_array=}')

        # Parameter loop
        for i, el in enumerate(parameters):
            type_len = parameter_lengths[i]
            if type_len is None:
                # Indicates a variable length string
                byte_array = file.read(4)
                assert len(byte_array) == 4
                type_len = int.from_bytes(byte_array, endianness)
                if not 0 <= type_len < 10000:
                    raise ValueError(
                        f'String length ({type_len}) ({byte_array}) is unreasonable - is file not {endianness}-endian?')
                if type_len > 0:
                    byte_array = file.read(type_len)
                    assert len(byte_array) == type_len
                    val = str(byte_array.decode('ascii'))
                else:
                    val = ''
            elif type_len == 1:
                byte_array = file.read(type_len)
                assert len(byte_array) == type_len
                val = np.frombuffer(byte_array, dtype=parameter_types[i], count=1)
                val = np.char.decode(val.view('S1'), 'ascii').astype(object)
                val = val[0]
            else:
                # All primitive types
                byte_array = file.read(type_len)
                assert len(byte_array) == type_len
                val = np.frombuffer(byte_array, dtype=parameter_types[i], count=1)[0]

            if TRACE:
                logger.debug(
                    f'>>PAR pos {file.tell()} | {parameter_types[i]} | {parameter_lengths[i]} | {type_len} | {val} | {byte_array}')

            # Assign data to the parameters
            if not page_skip:
                if TRACE:
                    logger.debug(f'{i}:{el}:{val}')
                el.data.append(val)

        # Array reading loop
        for i in range(n_arrays):
            a = sdds.arrays[i]
            flag = arrays_mask[i] and not page_skip
            type_len = arrays_size[i]
            mapped_t = arrays_type[i]
            if TRACE:
                logger.debug(f'>ARRAY {a.name} | {file.tell()=}')

            # Array dimensions
            byte_array = file.read(4 * a.dimensions)
            dimensions = np.frombuffer(byte_array, dtype=length_dtype)
            if len(dimensions) != a.dimensions:
                raise ValueError(
                    f'>>Array {a.name} dimensions {byte_array}/{dimensions} did not match expected count {a.dimensions}')
            n_elements = np.prod(dimensions)
            if TRACE:
                logger.debug(f'>>Array {a.name} | dimensions {dimensions}, total of {n_elements}')
            if n_elements > 1e7:
                raise Exception(f'Array is too large - {n_elements}')

            if type_len is None:
                # Strings need special treatment
                data_array = np.empty(dimensions, dtype=object) if flag else None
                for j in range(n_elements):
                    byte_array = file.read(4)
                    assert len(byte_array) == 4
                    string_len_actual = int.from_bytes(byte_array, endianness)
                    assert 0 <= string_len_actual <= 10000
                    if string_len_actual == 0:
                        if flag:
                            data_array[j] = ''
                    else:
                        if flag:
                            data_array[j] = file.read(string_len_actual).decode('ascii')
                if flag:
                    arrays[i].data.append(data_array)
            else:
                # Should read the right number of bytes or EOF
                data_bytes = file.read(type_len * n_elements)
                if len(data_bytes) < type_len * n_elements:
                    raise ValueError(f'>>Array {a.name} read failed because of EOF')
                if flag:
                    # Arrays are initialized in C order by default, matching SDDS
                    values = np.frombuffer(data_bytes, dtype=mapped_t)
                    #data_array = np.empty(dimensions, dtype=mapped_t)
                    #data_array[:] = values[:]
                    arrays[i].data.append(values.reshape(dimensions))

        # Column reading loop
        fixed_rowcount_eof = False
        if sdds.data.column_major_order != 0:
            # Column major order
            for i in range(n_columns):
                type_len = columns_len[i]
                flag = columns_mask[i] and not page_skip
                column_array = None
                if type_len is None:
                    if flag:
                        column_array = np.empty(page_size, dtype=object)
                    # string column
                    for row in range(page_size):
                        if TRACE:
                            logger.debug(f'>CMO COL {i} ROW {row} | {file.tell()=}')
                        byte_array = file.read(4)
                        assert len(byte_array) == 4
                        string_len_actual = int.from_bytes(byte_array, endianness)
                        assert 0 <= string_len_actual <= 10000  # sanity check
                        if string_len_actual == 0:
                            # empty string
                            if flag:
                                column_array[row] = ''
                                # l.debug(f'>>COL S {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                        else:
                            byte_array = file.read(string_len_actual)
                            if flag:
                                column_array[row] = byte_array.decode('ascii')
                                # l.debug(f'>>COL S {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                else:
                    # primitive type -> read in full column
                    byte_array = file.read(type_len * page_size)
                    if flag:
                        column_array = np.frombuffer(byte_array, dtype=columns_type[i], count=page_size)
                        if type_len == 1:
                            # Decode uint8 to <U1 to object
                            column_array = np.char.decode(column_array.view('S1'), 'ascii').astype(object)
                        # l.debug(f'>>COL {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                if TRACE:
                    logger.debug(f'>COL {i} END | {file.tell()=}')

                # Assign data
                if flag:
                    sdds.columns[i].data.append(column_array)
                    sdds.columns[i]._page_numbers.append(page_idx)

            if not page_skip:
                page_stored_idx += 1
        elif columns_all_numeric and sdds._meta_fixed_rowcount:
            for i in range(n_columns):
                if columns_mask[i]:
                    columns_data.append(np.empty(page_size, dtype=columns_store_type[i]))
            st = struct.Struct(combined_struct)
            page_size_actual = None
            for row in range(page_size):
                byte_array = file.read(combined_size)
                if len(byte_array) < combined_size:
                    if sdds._meta_fixed_rowcount:
                        logger.info(f'Encountered fixed rowcount file end at row {row} of {page_size}')
                        page_size_actual = row
                        fixed_rowcount_eof = True
                        if len(byte_array) > 0:
                            logger.warning(f'Weird leftover {repr(byte_array)}, ignoring')
                        break
                    else:
                        raise ValueError(f'Unexpected EOF at row {row}')
                if not page_skip:
                    values = st.unpack(byte_array)
                    idx_active = 0
                    for i, v in enumerate(values):
                        if columns_mask[i]:
                            columns_data[idx_active][row] = v
                            idx_active += 1
                if fixed_rowcount_eof:
                    break

            # Assign data to the columns
            if not page_skip:
                idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        if sdds._meta_fixed_rowcount and page_size_actual < page_size:
                            # Hopefully no copy?
                            arr = columns_data[idx_active][:page_size_actual]
                            if columns_len[i] == 1:
                                c.data.append(np.char.decode(arr.view('S1'), 'ascii').astype(object))
                            else:
                                c.data.append(arr)
                        else:
                            arr = columns_data[idx_active]
                            if columns_len[i] == 1:
                                c.data.append(np.char.decode(arr.view('S1'), 'ascii').astype(object))
                            else:
                                c.data.append(arr)
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1
        elif columns_all_numeric and not sdds._meta_fixed_rowcount and sdds._source_file_size > 500e6:
            for i in range(n_columns):
                if columns_mask[i]:
                    columns_data.append(np.empty(page_size, dtype=columns_store_type[i]))
            st = struct.Struct(combined_struct)
            byte_array = file.read(combined_size * page_size)
            if len(byte_array) < combined_size * page_size:
                raise ValueError(f'Unexpected EOF - got {len(byte_array)} bytes, wanted {combined_size * page_size}')
            if not page_skip:
                for row, tp in enumerate(st.iter_unpack(byte_array)):
                    idx_active = 0
                    for i, v in enumerate(tp):
                        if columns_mask[i]:
                            columns_data[idx_active][row] = v
                            idx_active += 1

                idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        arr = columns_data[idx_active]
                        if columns_len[i] == 1:
                            c.data.append(np.char.decode(arr.view('S1'), 'ascii').astype(object))
                        else:
                            c.data.append(arr)
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1
        elif columns_all_numeric and not sdds._meta_fixed_rowcount:
            if (combined_size * page_size) % combined_dtype.itemsize != 0:
                raise ValueError(f'Type length mismatch: {combined_size=} {page_size=} {combined_size*page_size=}'
                                 f' {combined_dtype.itemsize=} {(combined_size * page_size) % combined_dtype.itemsize=}')

            byte_array = file.read(combined_size * page_size)
            if len(byte_array) != combined_size * page_size:
                raise ValueError(f'Unexpected EOF - got {len(byte_array)} bytes, wanted {combined_size * page_size}')

            if not page_skip:
                array = np.frombuffer(byte_array, combined_dtype)
                idx_active = 0
                for i, c in enumerate(columns):
                    if columns_mask[i]:
                        # arr = array[f'f{i}'].copy()
                        if columns_len[i] == 1:
                            # c.data.append(np.char.decode(arr.view('S1'), 'ascii'))
                            c.data.append(np.char.decode(array[str(i)].view('S1'), 'ascii').astype(object))
                        else:
                            # For now, make a copy to be safe
                            c.data.append(array[str(i)].copy())
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1
        else:
            for i in range(n_columns):
                if columns_mask[i]:
                    columns_data.append(np.empty(page_size, dtype=columns_store_type[i]))
            page_size_actual = None
            for row in range(page_size):
                idx_active = 0
                if TRACE:
                    logger.debug(f'>COL ROW {row} | {file.tell()=}')
                for i in range(n_columns):
                    type_len = columns_len[i]
                    mapped_t = columns_type[i]
                    flag = columns_mask[i]
                    if type_len is None:
                        # string column
                        byte_array = file.read(4)
                        if len(byte_array) < 4:
                            if sdds._meta_fixed_rowcount:
                                logger.info(f'Encountered fixed rowcount file end at row {row} of {page_size}')
                                page_size_actual = row
                                fixed_rowcount_eof = True
                                if len(byte_array) > 0:
                                    raise ValueError(f'Weird leftover {byte_array}')
                                break
                            else:
                                raise ValueError(f'Unexpected EOF at row {row}, column {i}')
                        string_len_actual = int.from_bytes(byte_array, endianness)
                        assert 0 <= string_len_actual <= 10000  # sanity check
                        if string_len_actual == 0:
                            # empty string
                            if flag and not page_skip:
                                columns_data[idx_active][row] = ''
                                # l.debug(f'>>COL S {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                                idx_active += 1
                        else:
                            byte_array = file.read(string_len_actual)
                            if flag and not page_skip:
                                columns_data[idx_active][row] = byte_array.decode('ascii')
                                # l.debug(f'>>COL S {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                                idx_active += 1
                    else:
                        # primitive type column
                        byte_array = file.read(type_len)
                        if len(byte_array) < type_len:
                            if sdds._meta_fixed_rowcount:
                                logger.info(f'Encountered fixed rowcount file end at row {row} of {page_size}')
                                page_size_actual = row
                                fixed_rowcount_eof = True
                                if len(byte_array) > 0:
                                    raise ValueError(f'Weird leftover {byte_array}')
                                break
                            else:
                                raise ValueError(f'Unexpected EOF at row {row}, column {i}')
                        if flag and not page_skip:
                            value = columns_structs[i].unpack(byte_array)[0]
                            # value = np.frombuffer(byte_array, dtype=mapped_t, count=1)[0]
                            if type_len == 1:
                                # Decode uint8 to <U1
                                # value = chr(int(value))
                                value = np.char.decode(np.array(value).view('S1'), 'ascii')
                            columns_data[idx_active][row] = value
                            # l.debug(f'>>COL {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                            idx_active += 1
                if TRACE:
                    logger.debug(f'>COL END {row} | {file.tell()=}')
                if fixed_rowcount_eof:
                    break

            # Assign data to the columns
            if not page_skip:
                idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        if sdds._meta_fixed_rowcount and page_size_actual < page_size:
                            c.data.append(columns_data[idx_active][:page_size_actual])
                        else:
                            c.data.append(columns_data[idx_active])
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1

            if TRACE:
                logger.debug(f'Page {page_idx} data copy finished')

        page_idx += 1

        if fixed_rowcount_eof:
            break

        next_byte = file.peek(1)
        if len(next_byte) > 0:
            # More data exists
            if pages_mask is not None and page_idx == len(pages_mask):
                logger.warning(f'Pages mask {pages_mask} is too short - have at least {len(next_byte)} extra bytes')
                break
        else:
            # End of file
            break

    if flip_bytes:
        # parameters should already be in native format
        for i, el in enumerate(sdds.arrays):
            if arrays_mask[i]:
                for j in range(len(el.data)):
                    el.data[j] = el.data[j].byteswap().newbyteorder()

        for i, c in enumerate(sdds.columns):
            if columns_mask[i]:
                for j in range(len(c.data)):
                    c.data[j] = c.data[j].byteswap().newbyteorder()

    sdds.n_pages = page_stored_idx


def _read_pages_ascii_mixed_lines(file: IO[bytes],
                                  sdds: SDDSFile,
                                  arrays_mask: List[bool],
                                  columns_mask: List[bool],
                                  pages_mask: List[bool]) -> None:
    """ Line by line numeric data parser for lines_per_row == 1 """

    parameters = sdds.parameters
    parameters_type = [_NUMPY_DTYPES[el.type] for el in parameters]
    n_parameters = len(parameters)
    logger.debug(f'Parameter types: {parameters_type}')

    arrays = sdds.arrays
    arrays_type = [_NUMPY_DTYPES[el.type] for el in arrays]
    n_arrays = len(arrays)
    logger.debug(f'Array types: {arrays_type}')

    columns = sdds.columns
    columns_type = [_NUMPY_DTYPES[el.type] for el in columns]
    columns_store_type = [_NUMPY_DTYPE_FINAL[el.type] for el in columns]
    pd_column_dict = {i: columns_store_type[i] if columns_store_type[i] != object else str for i in
                      range(len(columns_type))}
    assert object in columns_type
    struct_type = None

    logger.debug(f'Column types: {columns_type}')
    logger.debug(f'struct_type: {struct_type}')

    page_idx = 0
    page_stored_idx = 0
    # Flag for eof since can't break out of two loops
    while True:
        if pages_mask is not None:
            if page_idx >= len(pages_mask):
                logger.debug(f'Reached last page {page_idx} in mask, have at least 1 more remaining but exiting early')
                break
            else:
                page_skip = pages_mask[page_idx]
        else:
            page_skip = False

        if page_skip:
            logger.debug(f'>>PG | pos {file.tell()} | skipping page {page_idx}')
        else:
            logger.debug(f'>>PG | pos %d | reading page %d', file.tell(), page_idx)

        # Read parameters
        par_idx = 0
        while par_idx < n_parameters:
            b_array = __get_next_line(file, strip=True)
            if b_array is None:
                raise Exception(f'>>PARS | pos {file.tell()} | unexpected EOF at page {page_idx}')
            if not page_skip:
                if parameters_type[par_idx] == object:
                    value = b_array.strip()
                    # Indicates a variable length string
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    if TRACE:
                        logger.debug(
                            f'>>PARS | pos {file.tell()} | {par_idx=} | {parameters_type[par_idx]} | {repr(b_array)} | {value}')
                else:
                    # Primitive types
                    value = np.fromstring(b_array, dtype=parameters_type[par_idx], sep=' ', count=1)[0]
                    if TRACE:
                        logger.debug(
                            f'>>PARV | pos {file.tell()} | {par_idx=} | {parameters_type[par_idx]} | {repr(b_array)} | {value}')
                parameters[par_idx].data.append(value)
            par_idx += 1

        # Read arrays
        array_idx = 0
        while array_idx < n_arrays:
            a = arrays[array_idx]
            mapped_t = arrays_type[array_idx]

            # Array dimensions
            b_array = __get_next_line(file, strip=True)
            if b_array is None:
                raise Exception(f'>>ARRS | pos {file.tell()} | unexpected EOF at page {page_idx}')

            dimensions = np.fromstring(b_array, dtype=int, sep=' ')
            n_elements = np.prod(dimensions)
            if len(dimensions) != a.dimensions:
                raise ValueError(f'>>Array {a.name} dimensions {b_array} did not match expected count {a.dimensions}')
            logger.debug(f'>>Array {a.name} has dimensions {dimensions}, total of {n_elements}')

            # Start reading array
            n_lines_read = 0
            n_elements_read = 0
            line_values = []
            if arrays_type[array_idx] == object:
                # Strings need special treatment
                while True:
                    b_array = __get_next_line(file).strip()
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f'>>ARRV | {file.tell()} | unexpected EOF at page {page_idx}')
                    values = shlex.split(b_array, posix=True)
                    logger.debug(
                        f'>>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {repr(b_array)} | {values} | {n_elements=} | {n_lines_read=}')
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f'Too many elements read during array parsing: {n_elements_read} (need {n_elements})')
            else:
                # Primitive types
                while True:
                    b_array = __get_next_line(file)
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f'>>ARRV | {file.tell()} | unexpected EOF at page {page_idx}')
                    values = np.fromstring(b_array, dtype=mapped_t, sep=' ', count=-1)
                    if TRACE:
                        logger.debug(
                            f'>>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {repr(b_array)} | {values} | {n_elements=} | {n_lines_read=}')
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f'Too many elements read during array parsing: {n_elements_read} (need {n_elements})')

            if arrays_mask[array_idx] and not page_skip:
                values = np.concatenate(line_values)
                # Arrays are initialized in C order by default, matching SDDS
                if mapped_t == str:
                    data_array = np.empty(dimensions, dtype=object)
                else:
                    data_array = np.empty(dimensions, dtype=mapped_t)
                data_array[:] = values[:]
                arrays[array_idx].data.append(data_array)
            array_idx += 1

        # Read column page size
        b_array = __get_next_line(file)
        if b_array is None:
            raise Exception(f'>>COLS | {file.tell()} | unexpected EOF at page {page_idx}')

        page_size = int(b_array)
        assert 0 <= page_size <= 1e7
        logger.debug(f'>>COLS | {file.tell()} | page {page_idx} size: {page_size}')

        # line = file.readline().decode('ascii')
        # list instead of generator to hopefully preallocate space
        if _ASCII_TEXT_PARSE_METHOD == 'read_table':
            # Because read_table will consume too much if allowed to touch file, have to copy out a single page
            # TODO: see if maybe wrapping file will have higher perf
            lines = [file.readline().decode('ascii') for i in range(page_size)]
            buf = io.StringIO('\n'.join((l for l in lines if not l.startswith('!'))))
            # buf = io.StringIO('\n'.join(lines))
            # lines = [file.readline() for i in range(page_size)]
            # buf = io.BytesIO(b''.join(lines))
            opts = dict(delim_whitespace=True, comment='!',
                        header=None, escapechar='\\',
                        nrows=page_size, skip_blank_lines=True,
                        skipinitialspace=True,
                        doublequote=False,
                        dtype=pd_column_dict,
                        engine='c',
                        low_memory=False,
                        na_filter=False,
                        na_values=None,
                        keep_default_na=False)
            # iowrap = io.TextIOWrapper(file, encoding='ascii')
            # df = pd.read_table(iowrap, **opts)
            # iowrap.detach()
            df = pd.read_table(buf, encoding='ascii', **opts)
            # df = pd.read_table(file, encoding='ascii', **opts)
            # print(df.dtypes)
            # Assign data to the columns
            if not page_skip:
                col_idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        c.data.append(df.loc[:, col_idx_active].values)
                        c._page_numbers.append(page_idx)
                        col_idx_active += 1
                page_stored_idx += 1
            page_idx += 1
        elif _ASCII_TEXT_PARSE_METHOD == 'shlex':
            columns_data = []
            if not page_skip:
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        columns_data.append(np.empty(page_size, dtype=columns_store_type[i]))
            for row in range(page_size):
                line = __get_next_line(file, accept_meta_commands=False, strip=True)
                if not page_skip:
                    line_len = len(line)
                    if line_len == 0:
                        raise ValueError(f'Unexpected empty string at position {file.tell()}')

                    col_idx_active = 0
                    col_idx = 0
                    values = shlex.split(line, posix=True)
                    if TRACE:
                        logger.debug(f'>COL ROW {row} | {len(values)}: {values=}')
                    for c in sdds.columns:
                        if columns_mask[col_idx]:
                            t = columns_type[col_idx]
                            if t == object:
                                value = values[col_idx]
                            else:
                                value = np.fromstring(values[col_idx], dtype=t, count=1, sep=' ')[0]
                            columns_data[col_idx_active][row] = value
                            if TRACE:
                                logger.debug(f'>>CR {row=} | {c.name}:{value}')
                            col_idx_active += 1
                        col_idx += 1

            # Assign data to the columns
            if not page_skip:
                col_idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        c.data.append(columns_data[col_idx_active])
                        c._page_numbers.append(page_idx)
                        col_idx_active += 1
                page_stored_idx += 1
            page_idx += 1
        # elif _ASCII_TEXT_PARSE_METHOD == 'state_machine':
        # # Columns data init
        # columns_data = []
        # if not page_skip:
        #     for i, c in enumerate(sdds.columns):
        #         if columns_mask[i]:
        #             columns_data.append(np.empty(page_size, dtype=columns_store_type[i]))
        #     # a simple state machine that seeks using two indices (start, end)
        #     col_idx = 0
        #     col_idx_active = 0
        #     pointer_last = 0
        #     if TRACE:
        #         logger.debug(f'>COL ROW {row}')
        #
        #     # Two state flag booleans (instead of an enum, for performance reasons)
        #     # True if inside quotes ("), and all characters should be treated literally, False otherwise
        #     is_literal_mode = False
        #     # Next character will be treated as escaped
        #     is_escape_mode = False
        #     value_contains_escape_sequences = False
        #     # True if scanning within a value, False if scanning the space between columns
        #     is_reading_spacing = True
        #
        #     for pointer in range(line_len):
        #         char = line[pointer]
        #         if char == '!':
        #             # everything afterwards should be ignored
        #             assert not is_literal_mode
        #             assert is_reading_spacing
        #             logger.debug(f'>>CR {pointer=} {row=} > {char=} COMMENT SKIP REST')
        #             break
        #         if TRACE:
        #             logger.debug(
        #                 f'>>CR {row=} | {col_idx=} | {col_idx_active=} | {pointer_last=} | {pointer=} | {char=}')
        #         if is_reading_spacing:
        #             if char == ' ':
        #                 # skip spaces
        #                 pointer_last = pointer
        #                 continue
        #             else:
        #                 # start reading next value
        #                 pointer_last = pointer
        #                 is_reading_spacing = False
        #
        #         if char == ' ' or pointer == line_len - 1:
        #             if is_escape_mode:
        #                 raise Exception
        #             if pointer == line_len - 1:
        #                 if is_literal_mode:
        #                     # Closing quote of line
        #                     assert char == '"'
        #                     is_literal_mode = False
        #                 # shift by one at end of string
        #                 pointer += 1
        #             if is_literal_mode:
        #                 # advance
        #                 continue
        #             else:
        #                 # end of value
        #                 # we should not be in literal mode
        #                 assert not is_literal_mode
        #                 # add to data if column in mask
        #                 if columns_mask[col_idx]:
        #                     value_str = line[pointer_last:pointer]
        #                     if columns_type[col_idx] == str:
        #                         if value_str.startswith('"') and value_str.endswith('"'):
        #                             value = value_str[1:-1]
        #                         else:
        #                             value = value_str
        #                         if value_contains_escape_sequences:
        #                             value = value.replace('\\"', '"')
        #                     else:
        #                         value = np.fromstring(value_str, dtype=columns_type[col_idx], count=1, sep=' ')[0]
        #                     columns_data[col_idx_active][row] = value
        #                     col_idx_active += 1
        #                     if TRACE:
        #                         logger.debug(
        #                             f'>>CR {row=} | {file.tell()} | {pointer_last=} | {pointer=} | {line[pointer_last:pointer]} | {value}')
        #                 else:
        #                     # l.debug(f'>>CR {row=} | {file.tell()} | {pointer_last=} | {pointer=} | {line[pointer_last:pointer]} | SKIP')
        #                     pass
        #                 is_reading_spacing = True
        #                 col_idx += 1
        #         elif char == '"':
        #             if not is_escape_mode:
        #                 # literal mode toggle
        #                 is_literal_mode = not is_literal_mode
        #             else:
        #                 is_escape_mode = False
        #                 continue
        #         elif char == '\\':
        #             if not is_escape_mode:
        #                 is_escape_mode = True
        #                 value_contains_escape_sequences = True
        #             else:
        #                 continue
        #         else:
        #             if is_escape_mode:
        #                 is_escape_mode = False
        #             # any other characted gets added to value
        #             continue

        # # Sanity checks
        # assert col_idx == len(sdds.columns)
        # assert 1 <= col_idx_active <= col_idx
        else:
            raise Exception(f'Unrecognized parse method: {_ASCII_TEXT_PARSE_METHOD}')

        next_byte = file.peek(1)
        if len(next_byte) > 0:
            # More data exists
            if pages_mask is not None and page_idx == len(pages_mask):
                logger.warning(f'Mask {pages_mask} ended but have at least {len(next_byte)} extra bytes - stopping')
                break
        else:
            # End of file
            break
    sdds.n_pages = page_stored_idx


def _read_pages_ascii_numeric_lines(file: IO[bytes],
                                    sdds: SDDSFile,
                                    arrays_mask: List[bool],
                                    columns_mask: List[bool],
                                    pages_mask: List[bool]) -> None:
    """ Line by line numeric data parser for lines_per_row == 1 """

    parameters = sdds.parameters
    parameter_types = [_NUMPY_DTYPES[el.type] for el in parameters]
    logger.debug(f'Parameter types: {parameter_types}')

    arrays = sdds.arrays
    arrays_type = [_NUMPY_DTYPES[el.type] for el in arrays]

    columns = sdds.columns
    columns_type = [_NUMPY_DTYPES[el.type] for el in columns]
    columns_store_type = [_NUMPY_DTYPE_FINAL[el.type] for el in columns]
    assert object not in columns_type
    struct_type = np.dtype(', '.join(columns_type))

    logger.debug(f'Column types: {columns_type}')
    logger.debug(f'struct_type: {struct_type}')

    page_idx = 0
    page_stored_idx = 0
    # Flag for eof since can't break out of two loops
    while True:
        if pages_mask is not None:
            if page_idx >= len(pages_mask):
                logger.debug(f'Reached last page {page_idx} in mask, have at least 1 more remaining but exiting early')
                break
            else:
                page_skip = pages_mask[page_idx]
        else:
            page_skip = False

        if page_skip:
            logger.debug(f'>>PG | pos {file.tell()} | skipping page {page_idx}')
        else:
            logger.debug(f'>>PG | pos %d | reading page %d', file.tell(), page_idx)

        # Read parameters
        parameter_data = []
        par_idx = 0
        par_line_num = 0
        while par_idx < len(parameter_types):
            b_array = __get_next_line(file)
            if b_array is None:
                raise Exception(f'>>PARS | pos {file.tell()} | unexpected EOF at page {page_idx}')
            par_line_num += 1

            if par_line_num > 10000:
                raise Exception('Did not finish parsing parameters after 10000 lines - something is wrong')

            if parameter_types[par_idx] == object:
                value = b_array.strip()
                # Indicates a variable length string
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if TRACE:
                    logger.debug(
                        f'>>PARS | pos {file.tell()} | {par_idx=} | {parameter_types[par_idx]} | {repr(b_array)} | {value}')
            else:
                # Primitive types
                value = np.fromstring(b_array, dtype=parameter_types[par_idx], sep=' ', count=1)[0]
                if TRACE:
                    logger.debug(
                        f'>>PARV | pos {file.tell()} | {par_idx=} | {parameter_types[par_idx]} | {repr(b_array)} | {value}')
            parameter_data.append(value)
            par_idx += 1

        # Assign data to the parameters
        if not page_skip:
            for i, el in enumerate(parameters):
                el.data.append(parameter_data[i])

        array_idx = 0
        while array_idx < len(arrays_type):
            a = sdds.arrays[array_idx]
            mapped_t = arrays_type[array_idx]

            # Array dimensions
            b_array = __get_next_line(file)
            if b_array is None:
                raise Exception(f'>>ARRS | pos {file.tell()} | unexpected EOF at page {page_idx}')

            dimensions = np.fromstring(b_array, dtype=int, sep=' ', count=-1)
            n_elements = np.prod(dimensions)
            if len(dimensions) != a.dimensions:
                raise ValueError(f'>>Array {a.name} dimensions {b_array} did not match expected count {a.dimensions}')
            logger.debug(f'>>Array {a.name} has dimensions {dimensions}, total of {n_elements}')

            # Start reading array
            n_lines_read = 0
            n_elements_read = 0
            line_values = []
            if arrays_type[array_idx] == object:
                # Strings need special treatment
                while True:
                    b_array = __get_next_line(file).strip()
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f'>>ARRV | {file.tell()} | unexpected EOF at page {page_idx}')
                    values = shlex.split(b_array, posix=True)
                    logger.debug(
                        f'>>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {repr(b_array)} | {values} | {n_elements=} | {n_lines_read=}')
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f'Too many elements read during array parsing: {n_elements_read} (need {n_elements})')
            else:
                # Primitive types
                while True:
                    b_array = __get_next_line(file)
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f'>>ARRV | {file.tell()} | unexpected EOF at page {page_idx}')
                    values = np.fromstring(b_array, dtype=mapped_t, sep=' ', count=-1)
                    if TRACE:
                        logger.debug(
                            f'>>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {repr(b_array)} | {values} | {n_elements=} | {n_lines_read=}')
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f'Too many elements read during array parsing: {n_elements_read} (need {n_elements})')

            if arrays_mask[array_idx] and not page_skip:
                values = np.concatenate(line_values)
                # Arrays are initialized in C order by default, matching SDDS
                if mapped_t == str:
                    data_array = np.empty(dimensions, dtype=object)
                else:
                    data_array = np.empty(dimensions, dtype=mapped_t)
                data_array[:] = values[:]
                arrays[array_idx].data.append(data_array)
            array_idx += 1

        # Read column page size
        b_array = __get_next_line(file)
        if b_array is None:
            raise Exception(f'>>COLS | {file.tell()} | unexpected EOF at page {page_idx}')

        page_size = int(b_array)
        assert 0 <= page_size <= 1e7
        logger.debug(f'>>COLS | {file.tell()} | page size: {page_size}')

        # line = file.readline().decode('ascii')
        # list instead of generator to hopefully preallocate space
        if _ASCII_NUMERIC_PARSE_METHOD == 'loadtxt':
            gen = (__get_next_line(file, accept_meta_commands=False, strip=True) for _ in range(page_size))
            if not page_skip:
                data = np.loadtxt(gen, dtype=struct_type, unpack=True, usecols=np.where(columns_mask)[0], comments='!')
                print(len(data), len(columns), len(columns_mask))
                col_idx_active = 0
                for col_idx, c in enumerate(columns):
                    if columns_mask[col_idx]:
                        c.data.append(data[col_idx_active])
                        c._page_numbers.append(page_idx)
                        col_idx_active += 1
                page_stored_idx += 1
            page_idx += 1
        elif _ASCII_NUMERIC_PARSE_METHOD == 'read_table':
            pd_column_dict = {i: columns_type[i] for i in range(len(columns_type))}
            lines = [file.readline().decode('ascii') for i in range(page_size)]
            buf = io.StringIO('\n'.join((l for l in lines if not l.startswith('!'))))
            # buf = io.StringIO('\n'.join(lines))
            # lines = [file.readline() for i in range(page_size)]
            # buf = io.BytesIO(b''.join(lines))
            opts = dict(delim_whitespace=True, comment='!',
                        header=None, escapechar='\\',
                        nrows=page_size, skip_blank_lines=True,
                        skipinitialspace=True,
                        doublequote=False,
                        dtype=pd_column_dict,
                        engine='c',
                        low_memory=False,
                        na_filter=False,
                        na_values=None,
                        keep_default_na=False)
            # iowrap = io.TextIOWrapper(file, encoding='ascii')
            # df = pd.read_table(iowrap, **opts)
            # iowrap.detach()
            df = pd.read_table(buf, encoding='ascii', **opts)
            # df = pd.read_table(file, encoding='ascii', **opts)
            # print(df.dtypes)
            # Assign data to the columns
            if not page_skip:
                col_idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        c.data.append(df.loc[:, col_idx_active].values)
                        c._page_numbers.append(page_idx)
                        col_idx_active += 1
                page_stored_idx += 1
            page_idx += 1

        next_byte = file.peek(1)
        if len(next_byte) > 0:
            # More data exists
            if pages_mask is not None and page_idx == len(pages_mask):
                logger.warning(f'Mask {pages_mask} ended but have at least {len(next_byte)} extra bytes - stopping')
                break
        else:
            # End of file
            break
    sdds.n_pages = page_stored_idx

