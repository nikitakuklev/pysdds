import _io
import copy
import csv
import io
import logging
import tempfile
import time
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import IO, List, Optional, Union

import numpy as np
import pandas as pd

from pysdds.structures import SDDSFile
from pysdds.util.constants import (
    _NUMPY_DTYPE_BE,
    _NUMPY_DTYPE_LE,
    _NUMPY_DTYPE_SIZES,
    _STRUCT_STRINGS_BE,
    _STRUCT_STRINGS_LE,
)
from pysdds.util.errors import SDDSWriteException

# The proper way to implement conditional logging is to check current level,
# but this creates too much overhead in hot loops. So, old school global vars it is.
logger = logging.getLogger(__name__)
DEBUG2 = False  # one more level from debug
TRACE = False  # two more levels from debug

NEWLINE_CHAR = "\n"

ARRAY_MAX_VALUES_PER_LINE = 10

_ASCII_TEXT_WRITE_METHOD = "sequential_python"


def write(
    sdds: SDDSFile,
    filepath: Union[Path, str, BytesIO],
    #          endianness: Optional[str] = 'auto',
    compression: Optional[str] = None,
    overwrite: Optional[bool] = False,
    use_best_settings: bool = False,
):
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
        If true, existing file at filepath will be overwritten if exists.
    use_best_settings : bool
        If true, override SDDS object settings like no_row_counts/row_major/etc. to
        optimize performance. ASCII/binary preference will be respected. It is recommended to
        keep this enabled.
    """
    assert isinstance(sdds, SDDSFile), "Data structure is not an SDDSFile!"

    if isinstance(filepath, str):
        filepath = Path(filepath)
    elif isinstance(filepath, (Path, io.IOBase, tempfile.SpooledTemporaryFile)):
        pass
    elif issubclass(filepath.__class__, _io.IOBase):
        pass
    else:
        raise Exception(f"Filepath type {type(filepath)} is not a string, Path, or BytesIO object")

    mode = sdds.mode

    if compression not in [None, "auto", "xz", "gz", "bz2"]:
        raise ValueError(f"SDDS compression ({compression}) is not recognized")

    sdds.validate_data()

    endianness = sdds.endianness

    if sdds.data.lines_per_row != 1:
        raise NotImplementedError("lines_per_row != 1 is not yet supported")

    if sdds.data.no_row_counts != 0:
        if mode == "binary":
            logger.debug(f"Ignoring {sdds.data.no_row_counts=} in binary mode")
        # assert mode == 'ascii', SDDSWriteException(f'Row count does not apply to binary files')

    logger.debug('Writing file to "%s"', str(filepath))
    # logger.info(f'Mode (%s), compression (%s), endianness (%s)', mode, compression, endianness)
    t_start = time.perf_counter()

    if isinstance(filepath, Path):
        file = _open_write_file(filepath, compression=compression, overwrite_ok=overwrite)
    else:
        # IO is already a stream
        file = _open_write_stream(filepath, compression=compression)

    if use_best_settings:
        sdds = copy.copy(sdds)
        sdds.data = copy.deepcopy(sdds.data)
        sdds.data.no_row_counts = 0
        sdds.data.lines_per_row = 1
        sdds.data.column_major_order = 1
        sdds.data.__use_best_settings = True

    _dump_header(sdds, file)
    # logger.info(f'Header write OK')

    if sdds.n_pages > 0:
        if mode == "ascii":
            _dump_data_ascii(sdds, file, best_settings=use_best_settings)
        else:
            _dump_data_binary(sdds, file, endianness)

    logger.debug(f"Written in {(time.perf_counter() - t_start) * 1e3:.3f} ms")
    # is_columns_numeric = not any(el.type == 'string' for el in sdds.columns)
    # logger.debug(f'Columns numeric: {is_columns_numeric}')


class WriterState(Enum):
    UNINITIALIZED = 0
    READY_FOR_NEXT_PAGE = 1
    WRITING_PAGE = 2
    CLOSED = 3


class IncrementalWriter:
    def __init__(
        self,
        sdds: SDDSFile,
        filepath: Union[Path, str, BytesIO],
        compression: Optional[str] = None,
        overwrite: Optional[bool] = False,
        use_best_settings: bool = False,
        write_method: Optional[str] = "fixed_rowcount",
        binary_fixed_rowcount: int = 1000000,
    ):
        assert isinstance(sdds, SDDSFile), "Data structure is not an SDDSFile!"

        if isinstance(filepath, str):
            filepath = Path(filepath)
        elif isinstance(filepath, (Path, io.IOBase, tempfile.SpooledTemporaryFile)):
            pass
        elif issubclass(filepath.__class__, _io.IOBase):
            pass
        else:
            raise Exception(f"Filepath type {type(filepath)} is not a string, Path, or BytesIO object")

        if compression not in [None, "auto", "xz", "gz", "bz2"]:
            raise ValueError(f"SDDS compression ({compression}) is not recognized")

        if sdds.data.lines_per_row != 1:
            raise NotImplementedError("lines_per_row != 1 is not yet supported")

        if sdds.data.no_row_counts != 0:
            if sdds.mode == "binary":
                logger.debug(f"Ignoring {sdds.data.no_row_counts=} in binary mode")

        self.compression = compression
        self.overwrite = overwrite
        self.use_best_settings = use_best_settings
        self.sdds = None
        self.file = None
        self.write_stage = WriterState.UNINITIALIZED
        self.binary_fixed_rowcount = binary_fixed_rowcount
        self.filepath = filepath
        self.mode = sdds.mode
        self.endianness = sdds.endianness
        self.write_method = write_method

        sdds = copy.copy(sdds)
        sdds.data = copy.deepcopy(sdds.data)

        if use_best_settings:
            sdds.data.nm["no_row_counts"] = 0
            sdds.data.nm["lines_per_row"] = 1
            sdds.data.__use_best_settings = True
        sdds.data.nm["column_major_order"] = 0
        if write_method == "fixed_rowcount":
            sdds._meta_fixed_rowcount = True
        else:
            sdds._meta_fixed_rowcount = False
        self.sdds = sdds

        self.current_page = 0

        if self.endianness == "big":
            NUMPY_DTYPE = _NUMPY_DTYPE_BE
            STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_BE
        else:
            NUMPY_DTYPE = _NUMPY_DTYPE_LE
            STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_LE
        self.NUMPY_DTYPE = NUMPY_DTYPE

        parameters = []
        p_types = []
        p_lengths: List[Optional[int]] = []
        for p in sdds.parameters:
            if p.fixed_value is None:
                parameters.append(p)
                t = p.type
                if t == "string":
                    p_types.append(None)
                    p_lengths.append(None)
                else:
                    p_types.append(self.NUMPY_DTYPE[t])
                    p_lengths.append(_NUMPY_DTYPE_SIZES[t])

        logger.debug(f"Parameters: {len(parameters)} of {len(sdds.parameters)}")
        logger.debug(f"Parameter types: {p_types}")
        logger.debug(f"Parameter lengths: {p_lengths}")
        self.p_types = p_types
        self.p_lengths = p_lengths

        columns = sdds.columns
        n_columns = len(columns)
        column_types = []
        column_lengths = []
        for i, c in enumerate(columns):
            t = c.type
            if t == "string":
                column_types.append(None)
                column_lengths.append(None)
            else:
                column_types.append(NUMPY_DTYPE[t])
                column_lengths.append(_NUMPY_DTYPE_SIZES[t])
        if n_columns > 0:
            logger.debug(f"Columns: {n_columns}")
            logger.debug(f"C types: {column_types}")
            logger.debug(f"C lengths: {column_lengths}")
        self.column_types = column_types
        self.column_lengths = column_lengths

        # self.begin()

    def begin(self):
        if self.write_stage != WriterState.UNINITIALIZED:
            raise Exception("Cannot begin writing again")

        logger.debug('Opening write stream to "%s"', str(self.filepath))
        logger.debug(
            "Mode (%s), compression (%s), endianness (%s)", self.sdds.mode, self.compression, self.sdds.endianness
        )
        t_start = time.perf_counter()

        if isinstance(self.filepath, Path):
            file = _open_write_file(self.filepath, compression=self.compression, overwrite_ok=self.overwrite)
        else:
            # IO is already a stream
            file = _open_write_stream(self.filepath, compression=self.compression)
        self.file = file
        _dump_header(self.sdds, file, ignore_fixed_rowcount=False)
        self.write_stage = WriterState.READY_FOR_NEXT_PAGE
        logger.debug(f"Header write done in {(time.perf_counter() - t_start) * 1e3:.3f} ms")

    def _write_str_binary(self, s: str):
        slen = len(s)
        len_bytes = slen.to_bytes(4, self.endianness)
        self.file.write(len_bytes)
        self.file.write(s.encode("ascii"))

    def _new_page_binary(self, parameter_data: List[Union[str, int, float]], array_data: List[np.ndarray]):
        file = self.file
        page_size = self.binary_fixed_rowcount
        page_bytes = page_size.to_bytes(4, self.endianness)
        file.write(page_bytes)

        for i, el in enumerate(self.sdds.parameters):
            type_len = self.p_lengths[i]
            if type_len is None:
                self._write_str_binary(parameter_data[i])
            elif type_len == 1:
                file.write(ord(parameter_data[i]).to_bytes(1, self.endianness))
            else:
                arr = np.array([parameter_data[i]], dtype=self.p_types[i])
                file.write(arr)

        for i, el in enumerate(sdds.arrays):
            # file.write(el.data[page_idx].shape.view(NUMPY_DTYPE['character'])])
            for d in array_data[i].shape:
                file.write(d.to_bytes(4, self.endianness))
            t = el.type
            if t == "string":
                for s in array_data[i]:
                    self._write_str_binary(s)
            elif t == "character":
                file.write(array_data[i].astype("S1").view(self.NUMPY_DTYPE["character"]))
            else:
                file.write(array_data[i].view(self.NUMPY_DTYPE[t]))

    def _new_page_binary_fixed_rowcount(
        self, parameter_data: List[Union[str, int, float]], array_data: List[np.ndarray]
    ):
        file = self.file
        page_size = self.binary_fixed_rowcount
        page_bytes = page_size.to_bytes(4, self.endianness)
        file.write(page_bytes)
        logger.debug(f"Starting page {self.current_page} with {page_size} rows in FIXED COUNT MODE")

        for i, el in enumerate(self.sdds.parameters):
            type_len = self.p_lengths[i]
            if type_len is None:
                self._write_str_binary(parameter_data[i])
            elif type_len == 1:
                file.write(ord(parameter_data[i]).to_bytes(1, self.endianness))
            else:
                arr = np.array([parameter_data[i]], dtype=self.p_types[i])
                file.write(arr)

        for i, el in enumerate(self.sdds.arrays):
            # file.write(el.data[page_idx].shape.view(NUMPY_DTYPE['character'])])
            for d in array_data[i].shape:
                file.write(d.to_bytes(4, self.endianness))
            t = el.type
            if t == "string":
                for s in array_data[i]:
                    self._write_str_binary(s)
            elif t == "character":
                file.write(array_data[i].astype("S1").view(self.NUMPY_DTYPE["character"]))
            else:
                file.write(array_data[i].view(self.NUMPY_DTYPE[t]))

    def _end_page_binary(self):
        if self.write_method == "fixed_rowcount":
            pass

    def end_page(self):
        if self.write_stage != WriterState.WRITING_PAGE:
            raise SDDSWriteException("Cannot close page when not writing one")
        logger.debug(f"Ending page {self.current_page}")

        if self.write_method == "fixed_rowcount":
            self._end_page_binary()

        self.write_stage = WriterState.READY_FOR_NEXT_PAGE

    def new_page(self, parameter_data: List[Union[str, int, float]] = None, array_data: List[np.ndarray] = None):
        """
        Start writing a new page. Must be called after writing the header, and before writing any data.
        Can be called again at any point afterward to close current page and start next one.
        """
        if self.write_stage not in [WriterState.READY_FOR_NEXT_PAGE, WriterState.WRITING_PAGE]:
            raise SDDSWriteException(f"Cannot write new page in current state {self.write_stage}")

        if self.write_stage == WriterState.WRITING_PAGE:
            if self.write_method == "fixed_rowcount":
                raise SDDSWriteException("Cannot write more than one page when using fixed_rowcount mode")
            else:
                logger.debug(f"Ending page {self.current_page} implicitly due to new page request")
                self.end_page()

        parameter_data = parameter_data or []
        array_data = array_data or []
        assert len(parameter_data) == len(self.sdds.parameters)
        assert len(array_data) == len(self.sdds.arrays)

        if self.mode == "ascii":
            raise NotImplementedError("Incremental writing is not supported in ASCII mode")

        if self.write_method == "fixed_rowcount":
            self._new_page_binary_fixed_rowcount(parameter_data, array_data)
        elif self.write_method == "seekable_page":
            raise NotImplementedError("Seekable page writing is not yet supported")
        elif self.write_method == "page_by_page":
            raise NotImplementedError("Variable rowcount writing is not yet supported")
            self._new_page_binary(parameter_data, array_data)

        self.current_page += 1
        self.write_stage = WriterState.WRITING_PAGE

    def write_rows(self, data_arrays: Union[list[np.ndarray], list[Union[str, float, int]]]):
        if self.write_stage != WriterState.WRITING_PAGE:
            raise Exception("Cannot write data before starting new page")
        if not all(isinstance(arr, np.ndarray) for arr in data_arrays):
            # assume this is a single row
            assert all(isinstance(x, (str, float, int)) for x in data_arrays), (
                f"Invalid data types in {data_arrays} for single rowrite"
            )
            data_arrays_np = [np.array([x], dtype=self.column_types[i]) for i, x in enumerate(data_arrays)]
            logger.debug(f"Single row {data_arrays} converted to  {data_arrays_np}")
        else:
            data_arrays_np = data_arrays
        lengths = [len(arr) for arr in data_arrays_np]
        if len(set(lengths)) != 1:
            raise ValueError("Data arrays must be of uniform length")

        if self.mode == "ascii":
            raise NotImplementedError("Incremental writing is not supported in ASCII mode")
        else:
            if self.write_method == "fixed_rowcount":
                self._write_rows_binary(data_arrays_np)

    def _write_rows_binary(self, data_arrays: list[np.ndarray]):
        """
        Write one or multiple rows to the current page. Data arrays must be in the same order as columns in the
        SDDSFile, and of uniform length.
        """
        n_rows = len(data_arrays[0])
        logger.debug(f"Writing {n_rows} rows to page {self.current_page}")

        page_data = []
        for i, el in enumerate(self.sdds.columns):
            t = el.type
            arr = data_arrays[i]
            if t == "string":
                logger.debug(f"Writing data {arr} with {t=} as string array")
                page_data.append(arr)
            elif t == "character":
                page_data.append(arr.astype("S1").view(dtype=self.NUMPY_DTYPE["character"]))
            else:
                logger.debug(f"Writing data {arr} with {t=} as view {self.column_types[i]=}")
                page_data.append(arr.view(dtype=self.column_types[i]))
            # print(t, el.data[page_idx], el.data[page_idx].dtype, page_data[-1])

        for row in range(n_rows):
            for i, el in enumerate(self.sdds.columns):
                t = el.type
                if t == "string":
                    self._write_str_binary(page_data[i][row])
                elif t == "character":
                    self.file.write(page_data[i][row])
                else:
                    self.file.write(page_data[i][row])

    def close(self):
        if self.write_stage not in [WriterState.READY_FOR_NEXT_PAGE, WriterState.WRITING_PAGE]:
            raise Exception("Cannot close in current state")

        if self.write_stage == WriterState.WRITING_PAGE:
            logger.debug(f"Ending page {self.current_page} implicitly due to close request")
            self.end_page()

        logger.debug("Closing write stream")
        self.file.close()
        self.write_stage = WriterState.CLOSED


def _open_write_stream(stream: BytesIO, compression: str = None):
    if compression is not None and compression not in ["xz", "gz", "bz2"]:
        raise ValueError(f"Compression format ({compression}) is not recognized")
    buffered_stream = stream
    try:
        if compression == "xz":
            import lzma

            stream = lzma.open(buffered_stream, "wb")
        elif compression == "gz":
            import gzip

            stream = gzip.open(buffered_stream, "wb")
        elif compression == "bz2":
            import bz2

            stream = bz2.open(buffered_stream, "wb")
        elif compression == "zip":
            import zipfile

            stream = zipfile.ZipFile(buffered_stream, "w")
        else:
            stream = buffered_stream
        if TRACE:
            logger.debug(f"File stream: {buffered_stream}")
            logger.debug(f"Final stream: {stream}")
        return stream
    except IOError as ex:
        logger.exception("Buffer IO failed")
        raise ex


def _open_write_file(filepath: Path, compression: str = None, overwrite_ok: bool = False):
    assert isinstance(filepath, Path)

    if filepath.exists():
        if not overwrite_ok:
            raise IOError(f"File path {filepath} already exists")
        else:
            logger.warning(f"File {filepath} will be overwritten")
    if filepath.is_dir():
        raise IOError(f"File path {filepath} is a directory, expect a file")
    if not filepath.parent.exists():
        raise IOError(f"Parent directory {filepath.parent} does not exist")
    # if not filepath.is_file():
    #    raise IOError(f'File ({filepath}) does not exist or cannot be read')

    if compression is not None and compression not in ["xz", "gz", "bz2"]:
        raise ValueError(f"Compression format ({compression}) is not recognized")

    try:
        buffered_stream = open(filepath, "wb", buffering=2097152)  # 2**20
        if compression == "xz":
            import lzma

            stream = lzma.open(buffered_stream, "wb")
        elif compression == "gz":
            import gzip

            stream = gzip.open(buffered_stream, "wb")
        elif compression == "bz2":
            import bz2

            stream = bz2.open(buffered_stream, "wb")
        elif compression == "zip":
            import zipfile

            stream = zipfile.ZipFile(buffered_stream, "w")
        else:
            stream = buffered_stream
        if TRACE:
            logger.debug(f"File stream: {buffered_stream}")
            logger.debug(f"Final stream: {stream}")
        return stream
    except IOError as ex:
        logger.exception(f"File {str(filepath)} IO failed")
        raise ex


def _dump_header(sdds: SDDSFile, file: IO[bytes], ignore_fixed_rowcount: bool = True):
    """Dump header to a string"""
    from .. import __version__

    def append(s):
        file.write((s + NEWLINE_CHAR).encode("ascii"))
        # line_buf.append(s)

    # Write version 5 by default to be safe
    append("SDDS5")

    # Write endianness meta-command if ascii
    if sdds.mode == "binary":
        append(f"!# {sdds.endianness}-endian")

    if not ignore_fixed_rowcount and sdds._meta_fixed_rowcount:
        append("!# fixed-rowcount")

    append(f"!Generated by pysdds {__version__}")
    append("!Submit issues at github.com/nikitakuklev/pysdds")

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


def _dump_data_ascii(sdds: SDDSFile, file: IO[bytes], best_settings):
    def append(s):
        file.write((s + NEWLINE_CHAR).encode("ascii"))

    def encode_if_needed(s):
        if len(s) == 0:
            return '""'
        result = ""
        flag = False
        for ch in s:
            if 32 <= ord(ch) < 127:
                if ch == " ":
                    flag = True
                    result += ch
                elif ch in ("\\", '"', "!"):
                    flag = True
                    result += "\\" + ch
                else:
                    result += ch
            else:
                if ord(ch) >= 127:
                    raise Exception(f"Non-ascii character {repr(ch)}")
                else:
                    result += f"\\{ord(ch):03o}"
                    flag = True
        if flag:
            return '"' + result + '"'
        else:
            return result

    def encode_char_if_needed(s):
        assert len(s) == 1
        if 32 <= ord(s) < 127:
            return s
        else:
            return f"\\{ord(s):03o}"

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

    if _ASCII_TEXT_WRITE_METHOD == "sequential_python":
        # Quoting needs to be handled carefully....
        opts = dict(
            header=False,
            index=False,
            mode="wb",
            encoding="utf-8",
            compression=None,
            line_terminator=NEWLINE_CHAR,
            quotechar='"',
            doublequote=False,
            escapechar="\\",
            # float_format='%.15e'
        )

        for page_idx in range(sdds.n_pages):
            if page_idx > 0 and sdds.data.no_row_counts == 1:
                # Must indicate new page if row count is missing
                append("")
            append(f"! page number {page_idx}")
            for j, p in enumerate(sdds.parameters):
                if p.fixed_value is None:
                    v = p.data[page_idx]
                    if p.type == "string":
                        append(encode_if_needed(v))
                    elif p.type == "double":
                        append(f"{v:.15e}")
                    elif p.type == "character":
                        append(encode_char_if_needed(v))
                    else:
                        append(str(v))

            for i, el in enumerate(sdds.arrays):
                data = el.data[page_idx]
                append(" ".join([str(i) for i in data.shape]) + f" ! {len(data.shape)}-dimensional array {el.name}")
                if el.type == "string":
                    sl = [encode_if_needed(v) for v in data]
                elif el.type == "double":
                    sl = [f"{v:.15e}" for v in data]
                else:
                    sl = [str(v) for v in data]
                append(" ".join(sl))

            if len(sdds.columns) > 0:
                page_size = len(sdds.columns[0].data[page_idx])
                if sdds.data.no_row_counts == 0:
                    append(str(page_size))
                for i in range(page_size):
                    sl = []
                    for j, c in enumerate(sdds.columns):
                        v = c.data[page_idx][i]
                        if c.type == "string":
                            sl.append(encode_if_needed(v))
                        elif c.type == "double":
                            sl.append(f"{v:.15e}")
                        else:
                            sl.append(str(v))
                    append(" ".join(sl))

    elif _ASCII_TEXT_WRITE_METHOD == "pandas":
        # Quoting needs to be handled carefully....
        opts = dict(
            header=False,
            index=False,
            mode="wb",
            encoding="ascii",
            compression=None,
            line_terminator=NEWLINE_CHAR,
            quotechar='"',
            doublequote=False,
            escapechar="\\",
            # float_format='%.15e'
        )

        param_df = sdds.parameters_to_df()
        for page_idx in range(sdds.n_pages):
            append(f"! page number {page_idx}")

            if len(sdds.parameters) > 0:
                param_df.iloc[page_idx, :].to_csv(file, **opts, sep="\n", quoting=csv.QUOTE_MINIMAL)

            for i, el in enumerate(sdds.arrays):
                data = el.data[page_idx]
                append(" ".join([str(i) for i in data.shape]) + f" ! {len(data.shape)}-dimensional array {el.name}")
                array_df = pd.DataFrame({"data": data}).T
                array_df.to_csv(file, **opts, sep=" ", quoting=csv.QUOTE_MINIMAL)

            if len(sdds.columns) > 0:
                page_size = len(sdds.columns[0].data[page_idx])
                append(str(page_size))

                # This creates issues with quoted numeric values after formatting
                # Also there are issues with escaping carriage returns
                df = sdds.columns_to_df(page_idx)
                df.to_csv(file, **opts, sep=" ", quoting=csv.QUOTE_NONNUMERIC)

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
    if endianness == "big":
        NUMPY_DTYPE = _NUMPY_DTYPE_BE
        STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_BE
    else:
        NUMPY_DTYPE = _NUMPY_DTYPE_LE
        STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_LE

    def _write_str(s: str):
        slen = len(s)
        len_bytes = slen.to_bytes(4, endianness)
        file.write(len_bytes)
        file.write(s.encode("ascii"))

    if sdds.n_pages == 0:
        raise Exception

    parameters = []
    p_types = []
    p_lengths: List[Optional[int]] = []
    for p in sdds.parameters:
        if p.fixed_value is None:
            parameters.append(p)
            t = p.type
            if t == "string":
                p_types.append(None)
                p_lengths.append(None)
            else:
                p_types.append(NUMPY_DTYPE[t])
                p_lengths.append(_NUMPY_DTYPE_SIZES[t])

    n_parameters = len(parameters)
    logger.debug(f"Parameters: {len(parameters)} of {len(sdds.parameters)}")
    logger.debug(f"Parameter types: {p_types}")
    logger.debug(f"Parameter lengths: {p_lengths}")

    arrays = sdds.arrays
    arrays_type = []
    arrays_size: List[Optional[int]] = []
    for i, a in enumerate(arrays):
        t = a.type
        if t == "string":
            mapped_t = object
        else:
            mapped_t = NUMPY_DTYPE[t]
        arrays_type.append(mapped_t)
        arrays_size.append(_NUMPY_DTYPE_SIZES[t])
    n_arrays = len(arrays_type)
    if n_arrays > 0:
        logger.debug(f"Arrays to parse: {len(arrays_type)}")
        logger.debug(f"Array types: {arrays_type}")
        logger.debug(f"Array lengths: {arrays_size}")

    columns = sdds.columns
    n_columns = len(columns)
    column_types = []
    column_lengths = []
    for i, c in enumerate(columns):
        t = c.type
        if t == "string":
            column_types.append(None)
            column_lengths.append(None)
        else:
            column_types.append(NUMPY_DTYPE[t])
            column_lengths.append(_NUMPY_DTYPE_SIZES[t])
    if n_columns > 0:
        logger.debug(f"Columns: {n_columns}")
        logger.debug(f"C types: {column_types}")
        logger.debug(f"C lengths: {column_lengths}")

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
            if t == "string":
                for s in el.data[page_idx]:
                    _write_str(s)
            elif t == "character":
                file.write(el.data[page_idx].astype("S1").view(NUMPY_DTYPE["character"]))
            else:
                file.write(el.data[page_idx].view(NUMPY_DTYPE[t]))

        page_data = []
        for i, el in enumerate(sdds.columns):
            t = el.type
            if t == "string":
                page_data.append(el.data[page_idx])
            elif t == "character":
                page_data.append(el.data[page_idx].astype("S1").view(dtype=NUMPY_DTYPE["character"]))
            else:
                page_data.append(el.data[page_idx].view(dtype=column_types[i]))
            # print(t, el.data[page_idx], el.data[page_idx].dtype, page_data[-1])
        for row in range(page_size):
            for i, el in enumerate(sdds.columns):
                t = el.type
                if t == "string":
                    _write_str(page_data[i][row])
                elif t == "character":
                    file.write(page_data[i][row])
                else:
                    file.write(page_data[i][row])
