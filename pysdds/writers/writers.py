import _io
import csv
import io
import logging
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import IO, List, Optional, Union

import pandas as pd

from pysdds.structures import *
from pysdds.util.constants import _NUMPY_DTYPE_BE, _NUMPY_DTYPE_LE, _NUMPY_DTYPE_SIZES, \
    _STRUCT_STRINGS_BE, _STRUCT_STRINGS_LE
from pysdds.util.errors import SDDSWriteException

# The proper way to implement conditional logging is to check current level,
# but this creates too much overhead in hot loops. So, old school global vars it is.
logger = logging.getLogger(__name__)
DEBUG2 = False  # one more level from debug
TRACE = False  # two more levels from debug

NEWLINE_CHAR = '\n'

ARRAY_MAX_VALUES_PER_LINE = 10

_ASCII_TEXT_WRITE_METHOD = 'sequential_python'


def write(sdds: SDDSFile,
          filepath: Union[Path, str, BytesIO],
          #          endianness: Optional[str] = 'auto',
          compression: Optional[str] = None,
          overwrite: Optional[bool] = False):
    """
    Parameters
    ----------
    sdds : SDDSFile
        A valid SDDSFile instance to be written
    filepath : str, path object, file-like object
        Output file path or stream implementing write() function. If None, a bytes array is returned.
    endianness : str
        Endianness to use for numerical values, one of 'auto', 'big', 'little'. Defaults to 'auto'.
    compression : str
        Compression to use when writing file. One of None, 'auto', 'xz', 'gz', 'bz2'. Defaults to None.
    overwrite : bool
        If true, existing file at filepath will be overwritten
    """
    assert isinstance(sdds, SDDSFile), "Data structure is not an SDDSFile!"

    if isinstance(filepath, str):
        filepath = Path(filepath)
    elif isinstance(filepath, (Path, io.IOBase, tempfile.SpooledTemporaryFile)):
        pass
    elif issubclass(filepath.__class__, _io.IOBase):
        pass
    else:
        raise Exception(f'Filepath type {type(filepath)} is not a string, Path, or BytesIO object')

    mode = sdds.mode

    if compression not in [None, 'auto', 'xz', 'gz', 'bz2']:
        raise ValueError(f'SDDS compression ({compression}) is not recognized')

    sdds.validate_data()

    endianness = sdds.endianness

    if sdds.data.lines_per_row != 1:
        raise NotImplementedError("lines_per_row != 1 is not yet supported")

    if sdds.data.no_row_counts != 0:
        assert mode == 'ascii', SDDSWriteException(f'Row count does not apply to binary files')

    logger.debug(f'Writing file to "%s"', str(filepath))
    #logger.info(f'Mode (%s), compression (%s), endianness (%s)', mode, compression, endianness)
    t_start = time.perf_counter()

    if isinstance(filepath, Path):
        file = _open_write_file(filepath, compression=compression, overwrite_ok=overwrite)
    else:
        # IO is already a stream
        file = _open_write_stream(filepath, compression=compression)

    _dump_header(sdds, file)
    #logger.info(f'Header write OK')

    if mode == 'ascii':
        _dump_data_ascii(sdds, file)
    else:
        _dump_data_binary(sdds, file, endianness)

    #logger.info(f'Finished in {(time.perf_counter() - t_start) * 1e3:.3f} ms')
    # is_columns_numeric = not any(el.type == 'string' for el in sdds.columns)
    # logger.debug(f'Columns numeric: {is_columns_numeric}')


def _open_write_stream(stream: BytesIO, compression: str = None):
    if compression is not None and compression not in ['xz', 'gz', 'bz2']:
        raise ValueError(f'Compression format ({compression}) is not recognized')
    buffered_stream = stream
    try:
        if compression == 'xz':
            import lzma
            stream = lzma.open(buffered_stream, 'wb')
        elif compression == 'gz':
            import gzip
            stream = gzip.open(buffered_stream, 'wb')
        elif compression == 'bz2':
            import bz2
            stream = bz2.open(buffered_stream, 'wb')
        elif compression == 'zip':
            import zipfile
            stream = zipfile.ZipFile(buffered_stream, 'w')
        else:
            stream = buffered_stream
        if TRACE:
            logger.debug(f'File stream: {buffered_stream}')
            logger.debug(f'Final stream: {stream}')
        return stream
    except IOError as ex:
        logger.exception(f'Buffer IO failed')
        raise ex


def _open_write_file(filepath: Path, compression: str = None, overwrite_ok: bool = False):
    assert isinstance(filepath, Path)

    if filepath.exists():
        if not overwrite_ok:
            raise IOError(f'File path {filepath} already exists')
        else:
            logger.warning(f'File {filepath} will be overwritten')
    if filepath.is_dir():
        raise IOError(f'File path {filepath} is a directory, expect a file')
    if not filepath.parent.exists():
        raise IOError(f'Parent directory {filepath.parent} does not exist')
    # if not filepath.is_file():
    #    raise IOError(f'File ({filepath}) does not exist or cannot be read')

    if compression is not None and compression not in ['xz', 'gz', 'bz2']:
        raise ValueError(f'Compression format ({compression}) is not recognized')

    try:
        buffered_stream = open(filepath, 'wb', buffering=2097152)  # 2**20
        if compression == 'xz':
            import lzma
            stream = lzma.open(buffered_stream, 'wb')
        elif compression == 'gz':
            import gzip
            stream = gzip.open(buffered_stream, 'wb')
        elif compression == 'bz2':
            import bz2
            stream = bz2.open(buffered_stream, 'wb')
        elif compression == 'zip':
            import zipfile
            stream = zipfile.ZipFile(buffered_stream, 'w')
        else:
            stream = buffered_stream
        if TRACE:
            logger.debug(f'File stream: {buffered_stream}')
            logger.debug(f'Final stream: {stream}')
        return stream
    except IOError as ex:
        logger.exception(f'File {str(filepath)} IO failed')
        raise ex


def _dump_header(sdds: SDDSFile, file: IO[bytes], ignore_fixed_rowcount: bool = True):
    """ Dump header to a string """
    from .. import __version__

    def append(s):
        file.write((s + NEWLINE_CHAR).encode('ascii'))
        # line_buf.append(s)

    # Write version 3 by default to be safe
    append('SDDS3')

    # Write endianness meta-command if ascii
    if sdds.mode == 'binary':
        append(f'!# {sdds.endianness}-endian')

    if not ignore_fixed_rowcount and sdds._meta_fixed_rowcount:
        append(f'!# fixed-rowcount')

    append(f'!Generated by pysdds {__version__}')
    append(f'!Submit issues at github.com/nikitakuklev/pysdds')

    # Start dumping namelists
    if sdds.description is not None:
        append(sdds.description.to_sdds())

    for el in sdds.parameters:
        append(el.to_sdds())

    for el in sdds.arrays:
        append(el.to_sdds())

    for el in sdds.columns:
        append(el.to_sdds())

    if sdds.data is not None:
        append(sdds.data.to_sdds())

    # final_string = NEWLINE_CHAR.join(line_buf) + NEWLINE_CHAR
    # return final_string


def _dump_data_ascii(sdds: SDDSFile, file: IO[bytes]):
    def append(s):
        file.write((s + NEWLINE_CHAR).encode('ascii'))

    def encode_if_needed(s):
        if len(s) == 0:
            return '\"\"'
        result = ''
        flag = False
        for ch in s:
            if 32 <= ord(ch) < 127:
                if ch == ' ':
                    flag = True
                    result += ch
                elif ch in ('\\', '"'):
                    flag = True
                    result += '\\' + ch
                else:
                    result += ch
            else:
                if ord(ch) >= 127:
                    raise Exception(f'Non-ascii character {repr(ch)}')
                else:
                    result += f'\\{ord(ch):03o}'
                    flag = True
        if flag:
            return '"' + result + '"'
        else:
            return result

    # if _ASCII_TEXT_WRITE_METHOD == 'sequential':
    #     # Quoting needs to be handled carefully....
    #     opts = dict(
    #         header=False,
    #         index=False,
    #         mode='wb',
    #         encoding='utf-8',
    #         compression=None,
    #         line_terminator=NEWLINE_CHAR,
    #         quotechar='"',
    #         doublequote=False,
    #         escapechar='\\',
    #         # float_format='%.15e'
    #     )
    #
    #     param_df = sdds.parameters_to_df()
    #     for page_idx in range(sdds.n_pages):
    #         append(f'! page number {page_idx}')
    #
    #         if len(sdds.parameters) > 0:
    #             param_df.iloc[page_idx, :].to_csv(file, **opts, sep='\n', quoting=csv.QUOTE_NONNUMERIC)
    #
    #         for i, el in enumerate(sdds.arrays):
    #             data = el.data[page_idx]
    #             append(' '.join([str(i) for i in data.shape]) + f' ! {len(data.shape)}-dimensional array {el.name}')
    #             array_df = pd.DataFrame({'data': data}).T
    #             array_df.to_csv(file, **opts, sep=' ', quoting=csv.QUOTE_MINIMAL)
    #
    #         if len(sdds.columns) > 0:
    #             page_size = len(sdds.columns[0].data[page_idx])
    #             append(str(page_size))
    #
    #             # Convert strings to escaped form
    #             df = sdds.columns_to_df(page_idx)
    #             for i, c in enumerate(sdds.columns):
    #                 if c.type == 'string':
    #                     df.iloc[:, i] = df.iloc[:, i].map(encode_if_needed)
    #
    #             for i in range(len(df)):
    #                 append(df.iloc[i:i + 1, :].to_string(index=False, header=False, float_format='%.15e'))

    if _ASCII_TEXT_WRITE_METHOD == 'sequential_python':
        # Quoting needs to be handled carefully....
        opts = dict(
            header=False,
            index=False,
            mode='wb',
            encoding='utf-8',
            compression=None,
            line_terminator=NEWLINE_CHAR,
            quotechar='"',
            doublequote=False,
            escapechar='\\',
            # float_format='%.15e'
        )

        for page_idx in range(sdds.n_pages):
            append(f'! page number {page_idx}')
            for j, p in enumerate(sdds.parameters):
                v = p.data[page_idx]
                if p.type == 'string':
                    append(encode_if_needed(v))
                elif p.type == 'double':
                    append(f'{v:.15e}')
                else:
                    append(str(v))

            for i, el in enumerate(sdds.arrays):
                data = el.data[page_idx]
                append(' '.join([str(i) for i in data.shape]) + f' ! {len(data.shape)}-dimensional array {el.name}')
                if el.type == 'string':
                    sl = [encode_if_needed(v) for v in data]
                elif el.type == 'double':
                    sl = [f'{v:.15e}' for v in data]
                else:
                    sl = [str(v) for v in data]
                append(' '.join(sl))

            if len(sdds.columns) > 0:
                page_size = len(sdds.columns[0].data[page_idx])
                if sdds.data.no_row_counts != 0:
                    append(str(page_size))
                for i in range(page_size):
                    sl = []
                    for j, c in enumerate(sdds.columns):
                        v = c.data[page_idx][i]
                        if c.type == 'string':
                            sl.append(encode_if_needed(v))
                        elif c.type == 'double':
                            sl.append(f'{v:.15e}')
                        else:
                            sl.append(str(v))
                    append(' '.join(sl))

    elif _ASCII_TEXT_WRITE_METHOD == 'pandas':
        # Quoting needs to be handled carefully....
        opts = dict(
            header=False,
            index=False,
            mode='wb',
            encoding='ascii',
            compression=None,
            line_terminator=NEWLINE_CHAR,
            quotechar='"',
            doublequote=False,
            escapechar='\\',
            # float_format='%.15e'
        )

        param_df = sdds.parameters_to_df()
        for page_idx in range(sdds.n_pages):
            append(f'! page number {page_idx}')

            if len(sdds.parameters) > 0:
                param_df.iloc[page_idx, :].to_csv(file, **opts, sep='\n', quoting=csv.QUOTE_MINIMAL)

            for i, el in enumerate(sdds.arrays):
                data = el.data[page_idx]
                append(' '.join([str(i) for i in data.shape]) + f' ! {len(data.shape)}-dimensional array {el.name}')
                array_df = pd.DataFrame({'data': data}).T
                array_df.to_csv(file, **opts, sep=' ', quoting=csv.QUOTE_MINIMAL)

            if len(sdds.columns) > 0:
                page_size = len(sdds.columns[0].data[page_idx])
                append(str(page_size))

                # This creates issues with quoted numeric values after formatting
                # Also there are issues with escaping carriage returns
                df = sdds.columns_to_df(page_idx)
                df.to_csv(file, **opts, sep=' ', quoting=csv.QUOTE_NONNUMERIC)

                # Slower way but hopefully works
                # df = sdds.columns_to_df(page_idx)
                # for c, col in zip(sdds.columns, df.columns):
                #     if c.type == 'string':
                #         df.loc[:, col] = df.loc[:, col].apply(lambda x: '"{}"'.format(x))
                #     elif c.type == 'double':
                #         df.loc[:, col] = df.loc[:, col].apply(lambda x: '{:.15e}'.format(x))
                # opts2 = dict(
                #     header=False,
                #     index=False,
                #     mode='wb',
                #     encoding='ascii',
                #     compression=None,
                #     line_terminator=NEWLINE_CHAR,
                #     quotechar=None,
                #     doublequote=False,
                #     escapechar='\\',
                #     #float_format='%.15e',
                # )
                # df.to_csv(file, **opts2, sep=' ', quoting=csv.QUOTE_NONE)
            # append('')


def _dump_data_binary(sdds: SDDSFile, file: IO[bytes], endianness):
    if endianness == 'big':
        NUMPY_DTYPE = _NUMPY_DTYPE_BE
        STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_BE
    else:
        NUMPY_DTYPE = _NUMPY_DTYPE_LE
        STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_LE

    def _write_str(s: str):
        slen = len(s)
        len_bytes = slen.to_bytes(4, endianness)
        file.write(len_bytes)
        file.write(s.encode('ascii'))

    if sdds.n_pages == 0:
        raise Exception

    parameters = []
    p_types = []
    p_lengths: List[Optional[int]] = []
    for p in sdds.parameters:
        if not p.fixed_value:
            parameters.append(p)
            t = p.type
            if t == 'string':
                p_types.append(None)
                p_lengths.append(None)
            else:
                p_types.append(NUMPY_DTYPE[t])
                p_lengths.append(_NUMPY_DTYPE_SIZES[t])

    logger.debug(f'Parameters: {len(parameters)} of {len(sdds.parameters)}')
    logger.debug(f'Parameter types: {p_types}')
    logger.debug(f'Parameter lengths: {p_lengths}')
    n_parameters = len(parameters)

    arrays = sdds.arrays
    arrays_type = []
    arrays_size: List[Optional[int]] = []
    for i, a in enumerate(arrays):
        t = a.type
        if t == 'string':
            mapped_t = object
        else:
            mapped_t = NUMPY_DTYPE[t]
        arrays_type.append(mapped_t)
        arrays_size.append(_NUMPY_DTYPE_SIZES[t])
    n_arrays = len(arrays_type)
    logger.debug(f'Arrays to parse: {len(arrays_type)}')
    logger.debug(f'Array types: {arrays_type}')
    logger.debug(f'Array lengths: {arrays_size}')

    for page_idx in range(sdds.n_pages):
        page_size = 0
        if len(sdds.columns) > 0:
            page_size = len(sdds.columns[0].data[page_idx])
            page_bytes = page_size.to_bytes(4, endianness)
            file.write(page_bytes)

        for i, el in enumerate(parameters):
            type_len = p_lengths[i]
            if type_len is None:
                _write_str(el.data[page_idx])
            elif type_len == 1:
                file.write(ord(el.data[page_idx]).to_bytes(1, endianness))
            else:
                file.write(el.data[page_idx])

        for i, el in enumerate(sdds.arrays):
            # file.write(el.data[page_idx].shape.view(NUMPY_DTYPE['character'])])
            for d in el.data[page_idx].shape:
                file.write(d.to_bytes(4, endianness))
            t = el.type
            if t == 'string':
                for s in el.data[page_idx]:
                    _write_str(s)
            elif t == 'character':
                file.write(el.data[page_idx].astype('S1').view(NUMPY_DTYPE['character']))
            else:
                file.write(el.data[page_idx].view(NUMPY_DTYPE[t]))

        page_data = []
        for el in sdds.columns:
            t = el.type
            if t == 'string':
                page_data.append(el.data[page_idx])
            elif t == 'character':
                page_data.append(el.data[page_idx].astype('S1').view(NUMPY_DTYPE['character']))
            else:
                page_data.append(el.data[page_idx].view(NUMPY_DTYPE[t]))
        for row in range(page_size):
            for i, el in enumerate(sdds.columns):
                t = el.type
                if t == 'string':
                    _write_str(page_data[i][row])
                elif t == 'character':
                    file.write(page_data[i][row])
                else:
                    file.write(page_data[i][row])
