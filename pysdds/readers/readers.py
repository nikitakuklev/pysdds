import csv
import io
import logging
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import IO, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pysdds.structures import Array, Associate, Column, Data, Description, Parameter, SDDSFile
from pysdds.util.constants import (
    _LONGDOUBLE_NATIVE,
    _NUMPY_DTYPE_BE,
    _NUMPY_DTYPE_FINAL,
    _NUMPY_DTYPE_LE,
    _NUMPY_DTYPE_SIZES,
    _NUMPY_DTYPES,
    _STRUCT_STRINGS_BE,
    _STRUCT_STRINGS_LE,
)

from ..util.conversions import LONGDOUBLE_FIELD_BYTES, decode_longdouble_fields
from ..util.errors import SDDSReadError
from .shlex_sdds import split_sdds
from .tokenizers import tokenize_namelist

# The proper way to implement conditional logging is to check current level,
# but this creates too much overhead in hot loops. So, old school global vars it is.
logger = logging.getLogger(__name__)
DEBUG2 = False  # one more level from debug
TRACE = False  # two more levels from debug
if TRACE:
    DEBUG2 = True
# Expected keys for various SDDS namelists
_KEYS_DESCRIPTION = {"text", "contents"}
_KEYS_PARAMETER = {
    "name",
    "symbol",
    "units",
    "description",
    "format_string",
    "type",
    "fixed_value",
}
_KEYS_ARRAY = {
    "name",
    "symbol",
    "units",
    "description",
    "format_string",
    "type",
    "group_name",
    "field_length",
    "dimensions",
}
_KEYS_COLUMN = {
    "name",
    "symbol",
    "units",
    "description",
    "format_string",
    "type",
    "field_length",
}
_KEYS_DATA = {
    "mode",
    "lines_per_row",
    "no_row_counts",
    "additional_header_lines",
    "column_major_order",
    "endian",
}
_KEYS_ASSOCIATE = {"filename", "path", "description", "contents", "sdds"}

_HEADER_PARSE_METHOD = "v2"

# _ASCII_TEXT_PARSE_METHOD = 'read_table'
_ASCII_TEXT_PARSE_METHOD = "shlex"
_ASCII_NUMERIC_PARSE_METHOD = "read_table"  # 'fromtxt'
# Pages with at most this many rows are parsed in plain Python: pandas.read_table has ~0.2-0.35 ms of fixed
# setup per call, which dominates for files made of many small pages. Measured crossover is ~800-1000 rows
# for 2-6 numeric columns (10-row pages: 185 us with pandas vs 7 us in Python).
_ASCII_SMALL_PAGE_ROWS = 1000


def _int_lenient(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        # e.g. "3.0" in an integer column - the previous prefix-scanning parser accepted this too
        return int(float(s))


def _ascii_cell_converter(numpy_type):
    """Python-level converter for one ASCII numeric cell of the given numpy dtype string"""
    dt = np.dtype(numpy_type)
    if dt.kind == "f":
        # float() would silently round longdouble to binary64
        return np.longdouble if dt.itemsize > 8 else float
    return _int_lenient


def _longdouble_from_bytes(buffer, endianness: str) -> np.ndarray:
    """Packed 16-byte longdouble fields -> np.longdouble array, or float64 where numpy has no 80-bit type"""
    if _LONGDOUBLE_NATIVE:
        return np.frombuffer(
            buffer, dtype=_NUMPY_DTYPE_BE["longdouble"] if endianness == "big" else _NUMPY_DTYPE_LE["longdouble"]
        )
    return decode_longdouble_fields(buffer, endianness)


def _split_ascii_row(line: str) -> List[str]:
    """Tokenize one ASCII data row. Rows without quotes or escapes (the vast majority) split on whitespace
    exactly like the lexer would, at a fraction of the cost."""
    if '"' in line or "\\" in line:
        return split_sdds(line, posix=True)
    return line.split()


def _decode_ascii_text_parameter(line: str, sdds_type: str) -> str:
    """Decode a string or character parameter from its ASCII data line"""
    value = line.strip()
    if len(value) == 1:
        # Single bare character (including \ or " themselves) is taken verbatim
        pass
    elif value.startswith('"') or "\\" in value:
        # Quoted or escaped - reuse the column lexer so \" \\ \! and octal escapes decode identically
        tokens = split_sdds(value, posix=True)
        value = tokens[0] if tokens else ""
    # Otherwise keep the whole line as-is (lenient towards unquoted strings with spaces)
    if sdds_type == "character" and len(value) != 1:
        raise SDDSReadError(f"Unrecognized character {value=}")
    return value


def _open_file(
    filepath: Path,
    compression: str,
    use_magic_values: bool = False,
    buffer_size: Optional[int] = None,
) -> (IO[bytes], int, bool):
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
        raise OSError(f"File ({filepath}) does not exist or cannot be read")

    # Check file size
    filesize = filepath.stat().st_size
    if filesize > 1e9:
        logger.warning(f"File size ({filesize / 1e9:.2f})MB is quite large and will cause performance issues")

    if compression is not None and compression not in ["auto", "xz", "gz", "bz2"]:
        raise ValueError(f"Compression format ({compression}) is not recognized")

    if compression == "auto":
        if use_magic_values:
            # https://www.garykessler.net/library/file_sigs.html
            magic_dict = {
                b"\xfd\x37\x7a\x58\x5a\x00": "xz",
                b"\x1f\x8b\x08": "gz",
                b"\x42\x5a\x68": "bz2",
                b"\x50\x4b\x03\x04": "zip",
            }
            max_len = max(len(x) for x in magic_dict)
            with open(filepath, "rb") as f:
                magic_bytes = f.read(max_len)
                match = None
                for magic, c in magic_dict.items():
                    if magic_bytes.startswith(magic):
                        match = c
                        break
                compression = match  # None->no compression
            logger.info(f"Auto compression resolved as ({compression}) from file byte signature")
        else:
            extension = filepath.suffix.strip(".")
            if extension in ["xz", "7z", "lzma"]:
                compression = "xz"
            elif extension in ["gz", "bz2", "zip"]:
                compression = extension
            else:
                compression = None
            if compression is not None:
                logger.debug(f"Auto compression resolved as ({compression}) from file extension")

    # By default, Python reads in small blocks of io.DEFAULT_BUFFER_SIZE = 8192
    # To encourage large reads and fully consuming small files, a large buffer is specified
    # For decompression, the decompress function is called on small blocks regardless, so we supply
    # a buffered reader to at least read large chunks from network storage
    # If file size is small enough, we try to buffer whole file immediately
    # See bpo-41486 for Python 3.10 patch that will improve decompress performance
    BUFSIZE = 8192 * 8  # 8388608
    if buffer_size == 0 and filesize < BUFSIZE:
        try:
            with open(filepath, "rb", buffering=BUFSIZE) as f:
                buf = f.read()
            if compression == "xz":
                import lzma

                buf_decompressed = lzma.decompress(buf)
            elif compression == "gz":
                import gzip

                buf_decompressed = gzip.decompress(buf)
            elif compression == "bz2":
                import bz2

                buf_decompressed = bz2.decompress(buf)
            else:
                buf_decompressed = buf
            stream_size = len(buf_decompressed)
            # BytesIO will do copy-on-write on 3.5+, which in our case means never
            # https://hg.python.org/cpython/rev/79a5fbe2c78f
            stream = io.BytesIO(buf_decompressed)
            # Need peak() method...this will do a copy...
            # Will avoid peaking during parsing, so ok to not buffer
            buffered_stream = stream  # io.BufferedReader(stream, buffer_size=filesize)
            if TRACE:
                logger.debug(f"File size {filesize} and parsing settings allow for full buffering")
                logger.debug(f"Final stream: {buffered_stream}")
            return buffered_stream, stream_size, False
        except OSError:
            logger.exception(f"File {filepath!s} memory-buffered IO failed")
            raise
    else:
        try:
            if buffer_size is None:
                buffered_stream = open(filepath, "rb")
            else:
                # If file was too large to ingest fully, fall back to standard buffer
                # Too large a value might be detrimental due to CPU caches
                sbuf = BUFSIZE if buffer_size == 0 else buffer_size  # 1048576 2**20
                buffered_stream = open(filepath, "rb", buffering=sbuf)
            if compression == "xz":
                import lzma

                stream = lzma.open(buffered_stream, "rb")
            elif compression == "gz":
                import gzip

                stream = gzip.open(buffered_stream, "rb")
            elif compression == "bz2":
                import bz2

                stream = bz2.open(buffered_stream, "rb")
            else:
                stream = buffered_stream
            if TRACE:
                logger.debug(f"File stream: {buffered_stream}")
                logger.debug(f"Final stream: {stream}")
            return stream, None, True
        except OSError:
            logger.exception(f"File {filepath!s} IO failed")
            raise


def read(
    filepath: Union[Path, str, IO[bytes]],
    pages: Optional[Iterable[int]] = None,
    arrays: Optional[Union[Iterable[int], Iterable[str]]] = None,
    cols: Optional[Union[Iterable[int], Iterable[str]]] = None,
    mode: Optional[str] = "auto",
    endianness: Optional[str] = "auto",
    compression: Optional[str] = "auto",
    header_only: Optional[bool] = False,
    allow_longdouble: Optional[bool] = False,
) -> SDDSFile:
    """
    Read in an SDDS file

    Parameters
    ----------
    filepath : str or Path or file-like object
        A valid absolute or relative file path, or a concrete Path object, or a byte stream (file, BytesIO, etc.)
    pages : array-like, optional
        If given, the list of page indices to read (starting from page 0). Note that page numbers in resulting SDDSFile
        will be renumbered in sequence. In other words, if pages=[0,2], valid indices for SDDSFile data would be page=0
        and page=1. This behaviour might change in the future.
    arrays : array-like, optional
        If given, the list of arrays to read. All entries must be either array indices or names.
    cols : array-like, optional
        If given, the list of columns to read. All entries must be either column indices or names.
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
    allow_longdouble: bool, optional
        Allow reading longdouble data type. Because of significant issues with handling this type in a portable manner,
        effort is made to support it on x86-64 linux only. No correctness guarantees are made, and on certain platforms
        silent precision loss or parsing failures might occur. Avoid longdouble if at all possible.

    Returns
    -------
    sdds_file: SDDSFile
        Instance containing the read data
    """

    # Argument verification
    if isinstance(filepath, str):
        filepath = Path(filepath)
    elif isinstance(filepath, (Path, io.IOBase)):
        pass
    else:
        raise Exception("Filepath is not a string or Path object")

    # Array consistency
    array_mask_mode = 0
    if arrays is not None:
        if all(isinstance(el, str) for el in arrays):
            array_mask_mode = 1
        elif all(isinstance(el, int) for el in arrays):
            array_mask_mode = 2
        else:
            raise ValueError(f"Array selection ({arrays}) is neither all strings nor all integer indices")

    # Column consistency
    column_mask_mode = 0
    if cols is not None:
        if all(isinstance(c, str) for c in cols):
            column_mask_mode = 1
        elif all(isinstance(c, int) for c in cols):
            column_mask_mode = 2
        else:
            raise ValueError(f"Column selection ({cols}) is neither all strings nor all integer indices")

    if mode not in ["auto", "binary", "ascii"]:
        raise ValueError(f"SDDS mode ({mode}) is not recognized")

    if endianness not in ["auto", "big", "little"]:
        raise ValueError(f"SDDS binary endianness ({endianness}) is not recognized")

    if compression not in [None, "auto", "xz", "gz", "bz2", "zip"]:
        raise ValueError(f"SDDS compression ({compression}) is not recognized")

    if pages is not None:
        try:
            assert all(isinstance(i, int) for i in pages)
            # We will not know how many pages are present until reading file, so create mask now
            pages_mask = [i in pages for i in range(max(pages) + 1)]
            pages = np.array(pages)
        except (AssertionError, TypeError, ValueError):
            raise ValueError("Pagelist is not an array-like object of ints") from None
    else:
        pages_mask = None

    sdds = SDDSFile()
    sdds._source_file = str(filepath)

    logger.debug('Opening file "%s"', str(filepath))
    # logger.debug(f'Mode (%s), compression (%s), endianness (%s)', mode, compression, endianness)
    if header_only:
        # Header needs a small read amount - use python default (typically 1 block, 8192)
        buffer_size = None
    elif cols is None and pages is None:
        # We are reading full file - try to buffer whole file aggressively
        # If too large, fallback with large buffer will be used
        buffer_size = 0
    else:
        # Use a middle-ground option
        # buffer_size = 1048576
        buffer_size = 262144  # 4 NFSV4 packets at max size

    t_start = time.perf_counter()
    if isinstance(filepath, Path):
        # File is opened in binary mode because it is necessary for data parsing,
        # reopening after header parsing would interfere with IO buffering
        file, file_size, peek_available = _open_file(
            filepath, compression, use_magic_values=False, buffer_size=buffer_size
        )
        sdds._source_file_size = file_size
        logger.debug(
            "Path opened as file in %.3f ms, size (%s)",
            (time.perf_counter() - t_start) * 1e3,
            file_size,
        )
    else:
        file = filepath
        try:
            file_size = len(filepath)
        except TypeError:
            file_size = None
        try:
            filepath.peek(1)
            peek_available = True
        except (TypeError, AttributeError):
            # Raw streams like io.BytesIO have no peek() - they get wrapped below
            peek_available = False
        sdds._source_file_size = 0
        logger.debug(
            "Path opened as byte stream in %.3f ms, size (%s)",
            (time.perf_counter() - t_start) * 1e3,
            file_size,
        )

    try:
        # First, read the header
        if _HEADER_PARSE_METHOD == "v2":
            _read_header_v2(file, sdds, mode, endianness)
        else:
            _read_header_fullstream(file, sdds, mode, endianness)
        logger.debug(
            f"Header parsed: {len(sdds.parameters)} parameters, {len(sdds.arrays)} arrays, {len(sdds.columns)} columns"
        )
        if TRACE:
            logger.debug(f"Params: {sdds.parameters}")
            logger.debug(f"Arrays: {sdds.arrays}")
            logger.debug(f"Columns: {sdds.columns}")
            logger.debug(f"Data: {sdds.data}")
            logger.debug(f"Description: {sdds.description}")

        if header_only:
            return sdds

        if sdds.data is None:
            # Header-only file (e.g. description and fixed-value parameters) - nothing else to parse
            return sdds

        # Handle the longdouble mess conservatively
        if (
            any(el.type == "longdouble" for el in sdds.parameters)
            or any(el.type == "longdouble" for el in sdds.arrays)
            or any(el.type == "longdouble" for el in sdds.columns)
        ):
            if not allow_longdouble:
                raise ValueError(
                    "Encountered longdouble data type, which is strongly discouraged. Override with "
                    '"allow_longdouble" to attempt parsing.'
                )
            elif _LONGDOUBLE_NATIVE:
                logger.debug("longdouble values will be read as np.longdouble (80-bit, padded to 16 bytes)")
            else:
                logger.warning(
                    f"np.longdouble is {np.dtype(np.longdouble).itemsize} bytes on this platform - "
                    "longdouble values will be converted to float64 (53-bit precision)"
                )
        if sdds._meta_fixed_rowcount:
            if sdds.mode != "binary":
                raise ValueError(f'Meta-command "!# fixed-rowcount" requires binary mode, not {sdds.mode}')

        # Verify that parameter, array, column, and page masks can be applied
        if array_mask_mode == 0:
            array_mask = [True for _ in range(len(sdds.arrays))]
        elif array_mask_mode == 1:
            available_set = set(arrays)
            parsed_names = sdds.array_names
            parsed_set = set(parsed_names)
            items_dict = sdds.array_dict
            if not available_set.issubset(parsed_set):
                raise ValueError(f"Requested arrays ({arrays}) are not a subset of available ({parsed_names})")
            array_mask = [el.name in available_set for el in sdds.arrays]
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
                raise ValueError(f"Requested columns ({cols}) are not a subset file data ({parsed_names})")
            column_mask = [c.name in available_set for c in sdds.columns]
            for c in parsed_set.difference(available_set):
                sdds.columns_dict[c]._enabled = False
        elif column_mask_mode == 2:
            column_mask = [False for _ in range(len(sdds.columns))]
            for cidx in set(range(len(sdds.columns))) - set(cols):
                sdds.columns[cidx]._enabled = False
            for column_index in cols:
                column_mask[column_index] = True
        else:
            raise ValueError

        if DEBUG2:
            logger.debug(f"Array mask: {array_mask}")
        if DEBUG2 or column_mask_mode != 0:
            logger.debug(f"Column mask: {column_mask}")
        if DEBUG2:
            logger.debug(f"Page mask: {pages_mask} (actual page count TBD)")

        # Character columns need the token-based parser too - they map to object dtype, not a numeric one
        is_columns_numeric = not any(el.type in ("string", "character") for el in sdds.columns)
        if DEBUG2:
            logger.debug(f"Columns numeric: {is_columns_numeric}")

        # Skip lines if necessary
        if sdds.data.additional_header_lines != 0:
            if sdds.mode == "binary":
                logger.warning('Option "additional_header_lines" will be ignored in binary mode')
            else:
                for i in range(sdds.data.additional_header_lines):
                    file.readline()

        # ASCII parser always detects end of file via peek(); binary one only when the stream size is unknown
        if not peek_available and (sdds.mode != "binary" or file_size is None):
            logger.debug("Wrapping stream in buffered reader since peek() was not available")
            file = io.BufferedReader(file)

        if sdds.mode == "binary":
            _read_pages_binary(
                file,
                sdds,
                file_size=file_size,
                arrays_mask=array_mask,
                columns_mask=column_mask,
                pages_mask=pages_mask,
            )
        else:
            # Streaming ascii data is not yet supported because performance is bad
            if sdds.data.lines_per_row != 1:
                raise NotImplementedError(f"lines_per_row = {sdds.data.lines_per_row} is not yet supported")

            if sdds.data.no_row_counts != 0:
                # suppressed warning - too annoying...yolo
                logger.debug("no_row_counts mode support is experimental, be careful")

            if is_columns_numeric:
                logger.debug(f"Calling ASCII numeric column parser with {_ASCII_NUMERIC_PARSE_METHOD=}")
                _read_pages_ascii_numeric_lines(
                    file,
                    sdds,
                    arrays_mask=array_mask,
                    columns_mask=column_mask,
                    pages_mask=pages_mask,
                )
            else:
                logger.debug(f"Calling ASCII mixed column parser with method {_ASCII_TEXT_PARSE_METHOD}")
                _read_pages_ascii_mixed_lines(
                    file,
                    sdds,
                    arrays_mask=array_mask,
                    columns_mask=column_mask,
                    pages_mask=pages_mask,
                )

        if pages is not None:
            if sdds.n_pages != len(pages):
                raise OSError(f"Parser failed - got {sdds.n_pages} pages while {len(pages)} were requested")
    finally:
        file.close()
        # Never let a pushed-back line leak into the next read() call
        pushback_line_buf.clear()

    cols_enabled = [c for c in sdds.columns if c._enabled]
    n_cols_enabled = len(cols_enabled)
    arrays_enabled = sum(1 for c in sdds.arrays if c._enabled)
    if len(cols_enabled) > 0:
        n_rows = sum(len(v) for v in cols_enabled[0].data)
    else:
        n_rows = 0
    # logger.info(f'Read finished in {(time.perf_counter() - t_start) * 1e3:.3f} ms')
    t_read = (time.perf_counter() - t_start) * 1e3
    logger.debug(
        f"Read in {t_read:.3f} ms, "
        f"{sdds.n_pages} pages, {n_rows} rows,"
        f" {len(sdds.parameters)} parameters,"
        f" {arrays_enabled}/{len(sdds.arrays)} arrays,"
        f" {n_cols_enabled}/{len(sdds.columns)} columns\n"
    )
    # logger.debug(f'File description:')
    logger.debug(f"{sdds.describe()}")
    return sdds


pushback_line_buf = deque()


def __get_next_line(
    stream: IO[bytes],
    accept_meta_commands: bool = True,
    cut_midline_comments: bool = True,
    cut_midline_without_quotes: bool = True,
    strip: bool = False,
    replace_tabs: bool = False,
) -> Optional[str]:
    """Find next line that has valid SDDS data"""

    def detab(s):
        # TODO: check if faster as a str.translate
        return s.replace("\t", " ") if replace_tabs else s

    line = None
    while True:
        if len(pushback_line_buf) > 0:
            line = pushback_line_buf.pop()
        if line is None:
            line = stream.readline().decode("ascii")
        if len(line) == 0:
            # EOF
            return None

        if "!" in line:
            # Even though both 'in' and 'find' operations are aliased to C code, using 'in' as initial check is
            # significantly faster (~10x), and makes sense since midpoint comments are expected to be rare
            ls_line = line.lstrip()
            if ls_line.startswith("!"):
                # Full comment line, most common case
                if ls_line.startswith("!#"):
                    # Meta-command
                    if accept_meta_commands:
                        return line
                    else:
                        raise ValueError(f"Meta-command {line} encountered unexpectedly")
                else:
                    # Regular comment
                    if TRACE:
                        logger.debug(f">>NXL | pos {stream.tell()} | SKIP FULL {line!r}")
                    line = None
                    continue
            else:
                if cut_midline_comments:
                    # Partial comment line, look for last !
                    idx = line.rfind("!")
                    # Only remove if not escaped
                    # TODO: recursively continue looking for more comments?
                    if line[idx - 1] != "\\":
                        if cut_midline_without_quotes and '"' in line[idx + 1 :]:
                            line_cut = line
                            if TRACE:
                                logger.debug(f">>NXL {stream.tell()} | NO CUT QUOTES {line!r} -> {line_cut!r}")
                        else:
                            line_cut = line[:idx]
                            if TRACE:
                                logger.debug(f">>NXL {stream.tell()} | CUT LINE {line!r} -> {line_cut!r}")
                    else:
                        line_cut = line
                        if TRACE:
                            logger.debug(f">>NXL {stream.tell()} | escaped comment, not cutting {line!r}")
                else:
                    line_cut = line
                if strip:
                    return detab(line_cut.strip())
                else:
                    return detab(line_cut)
        else:
            if strip:
                return detab(line.strip())
            else:
                return detab(line)


def _parse_namelist_entry(line: str, pos: int) -> Tuple[str, str, int]:
    """Parse one `key=value,` entry of a header namelist starting at line[pos].

    Returns the key, the value with surrounding quotes removed, and the position just past the trailing comma.
    """
    len_line = len(line)
    tokens = []
    no_chars_allowed = False
    while True:
        # logger.debug(f'Char {line[pos]} | tokens {tokens} | nca {no_chars_allowed}')
        if line[pos] == " ":
            no_chars_allowed = True
        elif line[pos] == "=":
            # Transition to value
            key = "".join(tokens)
            break
        else:
            if no_chars_allowed:
                raise Exception
            else:
                tokens.append(line[pos])
        pos += 1
        if pos >= len_line:
            raise Exception("End of line reached")
    if key == "":
        raise Exception

    pos += 1
    no_chars_allowed = False
    literal_mode = False
    value_tokens = []
    while True:
        c = line[pos]
        # logger.debug(f'{pos} | char {c} | nca {no_chars_allowed} | l {literal_mode} | tokens {value_tokens}')
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
            if c == " ":
                # At end of namelist
                no_chars_allowed = True
            elif c == '"':
                # Toggle literal mode
                literal_mode = True
            elif c == ",":
                value = "".join(value_tokens)
                pos += 1
                break
            else:
                if no_chars_allowed:
                    raise Exception
                else:
                    value_tokens.append(c)
        pos += 1
        if pos >= len_line:
            raise Exception(f"End of line reached | {pos} | char {c} | tokens {value_tokens} | nca {no_chars_allowed}")

    if literal_mode:
        raise Exception

    # l.debug(f'>>NM key: {key}')
    # l.debug(f'>>NM value: {value}')
    return key, value, pos


def _read_header_fullstream(file: IO[bytes], sdds: SDDSFile, mode: str, endianness: str) -> None:
    """
    Read SDDS header - the ASCII text that described the data contained in the SDDS file. This parser uses the common
    approach of ingesting a stream of bytes and checking tokens, meaning it works on more input types but is slower
    than using native python split/find/etc. functions
    """
    if DEBUG2:
        logger.debug("Parsing header with streaming reader")
    version_line = file.readline(10).decode("ascii").rstrip()
    line_num = 1
    if version_line[:4] != "SDDS" or len(version_line) != 5:
        raise AttributeError(f"Header parsing failed on line {line_num}: {version_line!r} is not a valid version")

    try:
        sdds_version = int(version_line[4])
    except ValueError:
        raise AttributeError(f"Unrecognized SDDS version: {version_line[4]!r}") from None

    if sdds_version > 5 or sdds_version < 1:
        raise ValueError(f"This package only supports SDDS version 5 or lower, file is version {sdds_version}")

    if DEBUG2:
        logger.debug(f"File version: {version_line}")

    namelists = []

    def __find_next_namelist(stream, accept_meta_commands=False):
        # accumulate multi-line parameters
        line = buffer = __get_next_line(
            stream,
            accept_meta_commands=accept_meta_commands,
            cut_midline_comments=False,
        )  # file.readline().decode('ascii')
        if accept_meta_commands and line.startswith("!#"):
            return buffer.strip()
        else:
            while not line.rstrip().endswith("&end"):
                line = __get_next_line(
                    stream, accept_meta_commands=False, cut_midline_comments=False
                )  # stream.readline().decode('ascii')
                logger.debug(f"Adding line [{line!r}] to multi-line namelist")
                if line is None:
                    raise SDDSReadError("Unexpected EOF during header parsing")
                buffer += line
            buffer2 = buffer.strip()
        return buffer2

    meta_endianness_set = False
    while True:
        line = __find_next_namelist(file, accept_meta_commands=True)
        # line = file.readline().decode('ascii').strip('\n')
        len_line = len(line)
        line_num += 1
        if TRACE:
            logger.debug("Line %d: %s", line_num, line)
        if line.startswith("!"):
            if not line.startswith("!#"):
                raise Exception(line)
            if line == "!# big-endian":
                if endianness == "auto" or endianness == "big":
                    sdds.endianness = "big"
                    logger.debug(f"Binary file endianness set to ({sdds.endianness}) from meta-command")
                    meta_endianness_set = True
                else:
                    raise ValueError(f"File endianness ({line}) does not match requested one ({endianness})")
            elif line == "!# little-endian":
                if endianness == "auto" or endianness == "little":
                    sdds.endianness = "little"
                    logger.debug(f"Binary file endianness set to ({sdds.endianness}) from meta-command")
                    meta_endianness_set = True
                else:
                    raise ValueError(f"File endianness ({line}) does not match requested one ({endianness})")
            elif line == "!# fixed-rowcount":
                logger.debug("Fixed rowcount meta-command found - setting flag")
                sdds._meta_fixed_rowcount = True
            else:
                raise Exception(f"Meta command {line} is not recognized")
            continue

        assert line[0] == "&"
        pos_end = line.find(" ")
        if pos_end == -1:
            raise Exception
        command = line[0:pos_end]

        # l.debug(f'>Command parse result: {command}')

        # Main nameloop iteration
        nm_dict = {}
        pos = pos_end
        while pos < len_line:
            if line[pos] == " ":
                # skip
                pos += 1
            elif line[pos] == "&":
                if line[pos + 1 : pos + 4] == "end":
                    # end of namelist
                    break
                else:
                    # unrecognized command end
                    raise Exception
            else:
                k, v, pos = _parse_namelist_entry(line, pos)
                nm_dict[k] = v

        logger.debug(">Parse result %s | %s", command, nm_dict)
        # hack around newlines in escaped strings
        if command == "&parameter":
            if "fixed_value" in nm_dict:
                nm_dict["fixed_value"] = nm_dict["fixed_value"].replace("\n", " ")

        nm_keys = set(nm_dict.keys())
        namelists.append(nm_dict)

        # expected_keys = None
        if command == "&description":
            if sdds.description is not None:
                raise ValueError("Duplicate description entry found")
            expected_keys = _KEYS_DESCRIPTION
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            sdds.description = Description(nm_dict)
        elif command == "&parameter":
            expected_keys = _KEYS_PARAMETER
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            sdds.parameters.append(Parameter(nm_dict, sdds=sdds))
        elif command == "&array":
            expected_keys = _KEYS_ARRAY
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            if "dimensions" in nm_keys:
                nm_dict["dimensions"] = int(nm_dict["dimensions"])
            if "field_length" in nm_keys:
                nm_dict["field_length"] = int(nm_dict["field_length"])
            sdds.arrays.append(Array(nm_dict, sdds=sdds))
        elif command == "&column":
            expected_keys = _KEYS_COLUMN
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            sdds.columns.append(Column(nm_dict))
        elif command == "&data":
            # This should be last command
            expected_keys = _KEYS_DATA
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            file_mode = nm_dict["mode"]
            if mode != "auto":
                assert mode == file_mode
            else:
                if not (file_mode == "binary" or file_mode == "ascii"):
                    raise Exception(f"Unrecognized mode ({file_mode}) found in file")
            if "endian" in nm_keys:
                data_endianness = nm_dict["endian"]
                if meta_endianness_set:
                    if data_endianness != sdds.endianness:
                        raise Exception("Mismatch of data and meta-command endianness")
                sdds.endianness = nm_dict["endian"]
                logger.debug(f"Binary file endianness set to ({sdds.endianness}) from data namelist")
            sdds.mode = file_mode
            sdds.data = Data(nm_dict)
            break
        elif command == "&include":
            raise ValueError("This package does not support &include namelist command")
        else:
            raise ValueError(f"Unrecognized namelist command {command} on line {line_num}")

    if sdds.columns or sdds.arrays or any(p.fixed_value is None for p in sdds.parameters):
        if sdds.data is None:
            raise SDDSReadError("SDDS file contains columns, arrays, or parameters - &data namelist is required")

    sdds.n_parameters = len(sdds.parameters)
    sdds.n_arrays = len(sdds.arrays)
    sdds.n_columns = len(sdds.columns)


translation_remove_endings = str.maketrans({"\r": " ", "\n": " "})


def _read_header_v2(file: IO[bytes], sdds: SDDSFile, mode: str, endianness: str) -> None:
    """
    Read SDDS header - the ASCII text that described the data contained in the SDDS file. This parser uses the common
    approach of ingesting a stream of bytes and checking tokens, meaning it works on more input types but is slower
    than using native python split/find/etc. functions
    """
    if DEBUG2:
        logger.debug("Parsing header with streaming reader")
    version_line = file.readline(10).decode("ascii").rstrip()
    namelist_idx = 1
    if version_line[:4] != "SDDS" or len(version_line) != 5:
        raise AttributeError(f"Header parsing failed on line {namelist_idx}: {version_line!r} is not a valid version")

    try:
        sdds_version = int(version_line[4])
    except ValueError:
        raise AttributeError(f"Unrecognized SDDS version: {version_line[4]!r}") from None

    if sdds_version > 5 or sdds_version < 1:
        raise ValueError(f"This package only supports SDDS version 5 or lower, file is version {sdds_version}")

    if DEBUG2:
        logger.debug(f"File version: {version_line}")

    namelists = []

    def __find_next_namelist(stream, accept_meta_commands=False):
        # accumulate multi-line parameters to find a complete namelist
        line = buffer = __get_next_line(
            stream,
            accept_meta_commands=accept_meta_commands,
            cut_midline_comments=False,
        )  # file.readline().decode('ascii')
        line_cnt = 0
        if line is None:
            # EOF - header ended without a &data namelist
            return None
        if accept_meta_commands and line.startswith("!#"):
            return buffer.strip()
        else:
            # buffer = buffer.translate(translation_remove_endings)
            while not line.rstrip().endswith("&end"):
                line = __get_next_line(
                    stream, accept_meta_commands=False, cut_midline_comments=False
                )  # stream.readline().decode('ascii')
                logger.debug("Adding line [%s] to multi-line namelist", repr(line))
                if line is None:
                    raise SDDSReadError("Unexpected EOF during header parsing")
                # sline = line.translate(translation_remove_endings)
                buffer += line
                line_cnt += 1
            buffer2 = buffer.strip()
        if TRACE:
            logger.debug("Found end of multi-line namelist after [%d] lines", line_cnt)
        return buffer2

    meta_endianness_set = False
    while True:
        line = __find_next_namelist(file, accept_meta_commands=True)
        if line is None:
            # No &data namelist - legal only if the file carries no per-page data (checked below)
            break
        # line = file.readline().decode('ascii').strip('\n')
        # len_line = len(line)
        namelist_idx += 1
        if TRACE:
            logger.debug("Namelist %d: [%s]", namelist_idx, repr(line))
        if line.startswith("!"):
            if not line.startswith("!#"):
                raise Exception(line)
            if line == "!# big-endian":
                if endianness == "auto" or endianness == "big":
                    sdds.endianness = "big"
                    logger.debug(f"Binary file endianness set to ({sdds.endianness}) from meta-command")
                    meta_endianness_set = True
                else:
                    raise ValueError(f"File endianness ({line}) does not match requested one ({endianness})")
            elif line == "!# little-endian":
                if endianness == "auto" or endianness == "little":
                    sdds.endianness = "little"
                    logger.debug(f"Binary file endianness set to ({sdds.endianness}) from meta-command")
                    meta_endianness_set = True
                else:
                    raise ValueError(f"File endianness ({line}) does not match requested one ({endianness})")
            elif line == "!# fixed-rowcount":
                logger.debug("Fixed rowcount meta-command found - setting flag")
                sdds._meta_fixed_rowcount = True
            else:
                raise Exception(f"Meta command {line} is not recognized")
            continue

        try:
            tags, keys, values = tokenize_namelist(line)
        except Exception as ex:  # hand-written tokenizer, any failure means a malformed namelist
            raise SDDSReadError(f'Failed to parse namelist from "{line!r}" ({ex=})') from ex

        assert len(tags) == 2, f"Invalid tags {tags}"
        assert tags[1] == "&end", f"Invalid tags {tags}"
        assert len(keys) == len(values), f"Keys and values do not match {keys} | {values}"
        command = tags[0]
        nm_dict = {k: v for k, v in zip(keys, values)}
        if DEBUG2:
            logger.debug(">Parse result %s | %s", command, nm_dict)

        # hack around newlines in escaped strings
        if command == "&parameter":
            if "fixed_value" in nm_dict:
                nm_dict["fixed_value"] = nm_dict["fixed_value"].replace("\r\n", " ").replace("\n", " ")

        nm_keys = set(nm_dict.keys())
        namelists.append(nm_dict)

        # expected_keys = None
        if command == "&description":
            if sdds.description is not None:
                raise ValueError("Duplicate description entry found")
            expected_keys = _KEYS_DESCRIPTION
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            sdds.description = Description(nm_dict)
        elif command == "&parameter":
            expected_keys = _KEYS_PARAMETER
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            sdds.parameters.append(Parameter(nm_dict, sdds=sdds))
        elif command == "&array":
            expected_keys = _KEYS_ARRAY
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            if "dimensions" in nm_keys:
                nm_dict["dimensions"] = int(nm_dict["dimensions"])
            if "field_length" in nm_keys:
                nm_dict["field_length"] = int(nm_dict["field_length"])
            sdds.arrays.append(Array(nm_dict, sdds=sdds))
        elif command == "&column":
            expected_keys = _KEYS_COLUMN
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            sdds.columns.append(Column(nm_dict))
        elif command == "&data":
            # This should be last command
            expected_keys = _KEYS_DATA
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            file_mode = nm_dict["mode"]
            if mode != "auto":
                assert mode == file_mode
            else:
                if not (file_mode == "binary" or file_mode == "ascii"):
                    raise Exception(f"Unrecognized mode ({file_mode}) found in file")
            if "lines_per_row" in nm_keys:
                nm_dict["lines_per_row"] = int(nm_dict["lines_per_row"])
            if "no_row_counts" in nm_keys:
                nm_dict["no_row_counts"] = int(nm_dict["no_row_counts"])
            if "column_major_order" in nm_keys:
                nm_dict["column_major_order"] = int(nm_dict["column_major_order"])
            if "endian" in nm_keys:
                data_endianness = nm_dict["endian"]
                if meta_endianness_set:
                    if data_endianness != sdds.endianness:
                        raise Exception("Mismatch of data and meta-command endianness")
                sdds.endianness = nm_dict["endian"]
                logger.debug(f"Binary file endianness set to ({sdds.endianness}) from data namelist")
            sdds.mode = file_mode
            sdds.data = Data(nm_dict)
            break
        elif command == "&include":
            raise ValueError("This package does not support &include namelist command")
        elif command == "&associate":
            expected_keys = _KEYS_ASSOCIATE
            if not nm_keys.issubset(expected_keys):
                raise AttributeError(f"Namelist keys {nm_keys} unexpected for namelist {command}")
            # warnings.warn(f'Associate list found - it will be parsed but otherwise ignored')
            sdds.associates.append(Associate(nm_dict))
        else:
            raise ValueError(f"Unrecognized namelist command {command} on line {namelist_idx}")

    if sdds.columns or sdds.arrays or any(p.fixed_value is None for p in sdds.parameters):
        if sdds.data is None:
            raise SDDSReadError("SDDS file contains columns, arrays, or parameters - &data namelist is required")

    sdds.n_parameters = len(sdds.parameters)
    sdds.n_arrays = len(sdds.arrays)
    sdds.n_columns = len(sdds.columns)


def _read_pages_binary(
    file: IO[bytes],
    sdds: SDDSFile,
    file_size: int,
    arrays_mask: List[bool],
    columns_mask: List[bool],
    pages_mask: Optional[List[bool]],
    convert_to_native_endianness: bool = True,
):
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
    if endianness == "big":
        NUMPY_DTYPE_STRINGS = _NUMPY_DTYPE_BE
        STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_BE
        if convert_to_native_endianness:
            flip_bytes = True
    elif endianness == "little":
        NUMPY_DTYPE_STRINGS = _NUMPY_DTYPE_LE
        STRUCT_DTYPE_STRINGS = _STRUCT_STRINGS_LE
    else:
        raise ValueError(f"SDDS endianness ({endianness}) is invalid")
    length_dtype = np.dtype(NUMPY_DTYPE_STRINGS["long"])

    # Assemble data types - parameters are first on every page, but those with fixed_value are skipped
    parameters = []
    parameter_types = []
    parameter_lengths: List[Optional[int]] = []
    for p in sdds.parameters:
        if p.fixed_value is None:
            parameters.append(p)
            t = p.type
            if t == "string":
                parameter_types.append(None)
                parameter_lengths.append(None)
            else:
                numpy_type = NUMPY_DTYPE_STRINGS[t]
                parameter_types.append(numpy_type)
                parameter_lengths.append(_NUMPY_DTYPE_SIZES[t])
    logger.debug("Parameters to parse: %d of %d", len(parameters), len(sdds.parameters))
    logger.debug("Parameter types: %s", parameter_types)
    logger.debug("Parameter lengths: %s", parameter_lengths)
    # n_parameters = len(parameters)

    # Arrays go here
    arrays = sdds.arrays
    arrays_type = []
    arrays_store_type = []
    arrays_size: List[Optional[int]] = []
    for i, a in enumerate(arrays):
        t = a.type
        if t == "string":
            mapped_t = object
        else:
            mapped_t = NUMPY_DTYPE_STRINGS[t]
        arrays_type.append(mapped_t)
        arrays_store_type.append(_NUMPY_DTYPE_FINAL[t])
        arrays_size.append(_NUMPY_DTYPE_SIZES[t])
    n_arrays = len(arrays_type)
    if n_arrays > 0:
        logger.debug("Arrays to parse: %d", len(arrays_type))
        logger.debug("Array types: %s", arrays_type)
        logger.debug("Array lengths: %s", arrays_size)

    # Columns follow arrays
    columns_type = []
    columns_type_struct = []
    columns_structs = []
    columns_store_type = []
    columns_len: List[Optional[int]] = []
    combined_struct = combined_size = combined_dtype = None
    columns = sdds.columns
    n_columns = len(columns)
    skip_all_columns = all(not v for v in columns_mask)
    for i, c in enumerate(columns):
        t = c.type
        columns_type.append(NUMPY_DTYPE_STRINGS[t])
        columns_store_type.append(_NUMPY_DTYPE_FINAL[t])
        columns_len.append(_NUMPY_DTYPE_SIZES[t])
        columns_type_struct.append(STRUCT_DTYPE_STRINGS[t])
        columns_structs.append(struct.Struct(STRUCT_DTYPE_STRINGS[t]) if t != "string" else None)
    columns_all_numeric = object not in columns_type and n_columns > 0
    if columns_all_numeric:
        # One explicit byte-order prefix, then the bare codes (longdouble is "16s", which has no prefix of its own)
        combined_struct = ("<" if endianness == "little" else ">") + "".join(
            v.lstrip("<>") for v in columns_type_struct
        )
        combined_size = sum(columns_len)
    if n_columns > 0:
        logger.debug("Columns to parse: %d", n_columns)
        logger.debug("Column types: %s", columns_type)
        logger.debug("Column lengths: %s", columns_len)
        logger.debug("All numeric: %s", columns_all_numeric)

    if sdds.data.column_major_order != 0:
        logger.debug("Data is in column-major order - no special handling required")
    elif columns_all_numeric and sdds._meta_fixed_rowcount:
        # Numeric types but fixed rows - have to parse row by row
        logger.debug("All columns numeric and data is row order -> reading whole rows")
    elif (
        columns_all_numeric
        and not sdds._meta_fixed_rowcount
        and sdds._source_file_size is not None
        and sdds._source_file_size > 500e6
    ):
        # Row by row parsing with single struct - memory efficient and fast
        logger.debug("All columns numeric, no fixed rows, data is row order, large size -> using row-wide struct")
    elif columns_all_numeric and not sdds._meta_fixed_rowcount:
        # Whole page parsing by using buffer as structured array - the fastest, zero copy method
        logger.debug("All columns numeric, no fixed rows, row order, small size -> using structured array")
        # must specify endianness, or linux/windows struct lengths will differ!!!
        combined_dtype = np.dtype([(str(i), c.descr[0][1]) for i, c in enumerate(columns_type)])
    else:
        # Most general row-order parser
        logger.debug("Not all columns numeric, data is row order -> using slow sequential parser")

    # Row plan for the general parser: each run of consecutive non-string columns, together with the 4-byte
    # length prefix of the string column that follows it, is read and unpacked in one go. This cuts the
    # per-row Python work from one read+unpack per cell to one per segment. Entries are
    # (Struct, byte size, [(tuple position, destination index, is_character)...], has_string, string destination)
    row_plan = []
    if not columns_all_numeric and sdds.data.column_major_order == 0:
        prefix = "<" if endianness == "little" else ">"
        dest = []  # destination (active) index per column, None if masked out
        for i in range(n_columns):
            dest.append(sum(columns_mask[:i]) if columns_mask[i] else None)
        fmt = ""
        seg_numeric = []
        for i in range(n_columns):
            if columns_len[i] is None:
                st = struct.Struct(prefix + fmt + "i")
                row_plan.append((st, st.size, seg_numeric, True, dest[i]))
                fmt, seg_numeric = "", []
            else:
                if dest[i] is not None:
                    seg_numeric.append((len(fmt.replace("16s", "x")), dest[i], columns_len[i] == 1))
                fmt += columns_type_struct[i].lstrip("<>")
        if fmt:
            st = struct.Struct(prefix + fmt)
            row_plan.append((st, st.size, seg_numeric, False, None))

    # Cell dtypes for the struct-based parsers: struct yields 1-byte bytes for characters and 16 raw bytes for
    # longdouble, so those are collected as S1/V16 and converted once per page in _finish_column
    columns_cell_dtype = []
    for i, c in enumerate(columns):
        if columns_len[i] == 1:
            columns_cell_dtype.append(np.dtype("S1"))
        elif c.type == "longdouble":
            columns_cell_dtype.append(np.dtype(f"V{LONGDOUBLE_FIELD_BYTES}"))
        else:
            columns_cell_dtype.append(columns_store_type[i])

    def _finish_column(i: int, arr: np.ndarray) -> np.ndarray:
        if columns_len[i] == 1:
            return np.char.decode(arr.view("S1"), "ascii").astype(object)
        if columns[i].type == "longdouble":
            return _longdouble_from_bytes(np.ascontiguousarray(arr).tobytes(), endianness)
        return arr

    page_idx = 0
    page_stored_idx = 0
    page_last_active_idx = max(i for i, v in enumerate(pages_mask) if v) if pages_mask is not None else None
    while True:
        columns_data = []

        # Main loop
        if pages_mask is not None:
            if page_idx >= len(pages_mask):
                raise Exception("Should not be reachable")
            else:
                page_skip = not pages_mask[page_idx]
        else:
            page_skip = False

        # Read page size (row count)
        # The C reference uses a signed int32 here. If the value is INT32_MIN (-2147483648),
        # it is a sentinel indicating that the actual row count follows as a signed int64.
        byte_array = file.read(4)
        if len(byte_array) == 0:
            # we have a blank file, abort
            break
        assert len(byte_array) == 4
        page_size = int.from_bytes(byte_array, endianness, signed=True)
        if page_size == -2147483648:  # INT32_MIN sentinel for 64-bit row count
            byte_array_64 = file.read(8)
            assert len(byte_array_64) == 8
            page_size = int.from_bytes(byte_array_64, endianness, signed=True)
            logger.debug("Page %d uses 64-bit row count: %d", page_idx, page_size)
        if page_size < 0:
            raise ValueError(f"Page size ({page_size}) ({byte_array}) is negative - file is likely corrupt")
        if page_size > 1e12:
            raise ValueError(
                f"Page size ({page_size}) ({byte_array}) is unreasonable - is file not {endianness}-endian?"
            )
        logger.debug("Page %d size is %d | byte_array=%s", page_idx, page_size, byte_array)

        # Parameter loop
        for i, el in enumerate(parameters):
            type_len = parameter_lengths[i]
            if type_len is None:
                # Indicates a variable length string
                byte_array = file.read(4)
                assert len(byte_array) == 4
                type_len = int.from_bytes(byte_array, endianness, signed=True)
                if type_len < 0:
                    raise ValueError(f"String length ({type_len}) ({byte_array}) is negative - file is likely corrupt")
                if type_len >= 1000000:
                    raise ValueError(
                        f"String length ({type_len}) ({byte_array}) too large - is file not {endianness}-endian?"
                    )
                if type_len > 0:
                    byte_array = file.read(type_len)
                    assert len(byte_array) == type_len
                    val = str(byte_array.decode("ascii"))
                else:
                    val = ""
            elif type_len == 1:
                byte_array = file.read(type_len)
                assert len(byte_array) == type_len
                val = np.frombuffer(byte_array, dtype=parameter_types[i], count=1)
                val = np.char.decode(val.view("S1"), "ascii").astype(object)
                val = val[0]
            else:
                # All primitive types
                byte_array = file.read(type_len)
                assert len(byte_array) == type_len, f"Invalid {len(byte_array)=} for type {parameter_types[i]}"
                if el.type == "longdouble":
                    val = _longdouble_from_bytes(byte_array, endianness)[0]
                else:
                    val = np.frombuffer(byte_array, dtype=parameter_types[i], count=1)[0]

            if TRACE:
                logger.debug(
                    f">>PAR pos {file.tell()} | {parameter_types[i]} | {parameter_lengths[i]} | {type_len} | {val} | {byte_array}"
                )

            # Assign data to the parameters
            if not page_skip:
                if TRACE:
                    logger.debug(f"{i}:{el}:{val}")
                el.data.append(val)

        # Array reading loop
        for i in range(n_arrays):
            a = sdds.arrays[i]
            flag = arrays_mask[i] and not page_skip
            type_len = arrays_size[i]
            mapped_t = arrays_type[i]
            if TRACE:
                logger.debug(f">ARRAY {a.name} | {file.tell()=}")

            # Array dimensions
            byte_array = file.read(4 * a.dimensions)
            dimensions = np.frombuffer(byte_array, dtype=length_dtype)
            if len(dimensions) != a.dimensions:
                raise ValueError(
                    f">>Array {a.name} dimensions {byte_array}/{dimensions} did not match expected count {a.dimensions}"
                )
            n_elements = np.prod(dimensions)
            if TRACE:
                logger.debug(f">>Array {a.name} | dimensions {dimensions}, total of {n_elements}")
            if n_elements > 1e7:
                raise Exception(f"Array is too large - {n_elements}")

            if type_len is None:
                # Strings need special treatment
                # Filled with a flat C-order index, reshaped to the declared dimensions once complete
                data_array = np.empty(n_elements, dtype=object) if flag else None
                for j in range(n_elements):
                    byte_array = file.read(4)
                    assert len(byte_array) == 4
                    string_len_actual = int.from_bytes(byte_array, endianness, signed=True)
                    if string_len_actual < 0:
                        raise ValueError(
                            f"Array string length ({string_len_actual}) is negative - file is likely corrupt"
                        )
                    assert string_len_actual <= 1000000
                    if string_len_actual == 0:
                        if flag:
                            data_array[j] = ""
                    else:
                        string_bytes = file.read(string_len_actual)
                        if flag:
                            data_array[j] = string_bytes.decode("ascii")
                if flag:
                    arrays[i].data.append(data_array.reshape(dimensions))
            else:
                # Should read the right number of bytes or EOF
                data_bytes = file.read(type_len * n_elements)
                if len(data_bytes) < type_len * n_elements:
                    raise ValueError(f">>Array {a.name} read failed because of EOF")
                if flag:
                    # Arrays are initialized in C order by default, matching SDDS
                    if a.type == "longdouble":
                        values = _longdouble_from_bytes(data_bytes, endianness)
                    else:
                        values = np.frombuffer(data_bytes, dtype=mapped_t)
                    if type_len == 1:
                        # Character arrays are stored as str objects, like character columns and parameters
                        values = np.char.decode(values.view("S1"), "ascii").astype(object)
                    arrays[i].data.append(values.reshape(dimensions))

        # Column reading loop
        fixed_rowcount_eof = False
        if n_columns == 0:
            if not page_skip:
                page_stored_idx += 1
        elif page_last_active_idx is not None and page_idx == page_last_active_idx and skip_all_columns:
            # end of file, don't need columns - terminate early
            logger.debug("Last active page with no columns to read - terminating early")
            if not page_skip:
                page_stored_idx += 1
            break
        elif sdds.data.column_major_order != 0:
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
                            logger.debug(f">CMO COL {i} ROW {row} | {file.tell()=}")
                        byte_array = file.read(4)
                        assert len(byte_array) == 4
                        string_len_actual = int.from_bytes(byte_array, endianness, signed=True)
                        if string_len_actual < 0:
                            raise ValueError(
                                f"Column string length ({string_len_actual}) is negative - file is likely corrupt"
                            )
                        assert string_len_actual <= 1000000  # sanity check
                        if string_len_actual == 0:
                            # empty string
                            if flag:
                                column_array[row] = ""
                                # l.debug(f'>>COL S {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                        else:
                            byte_array = file.read(string_len_actual)
                            if flag:
                                column_array[row] = byte_array.decode("ascii")
                                # l.debug(f'>>COL S {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                else:
                    # primitive type -> read in full column
                    byte_array = file.read(type_len * page_size)
                    if flag:
                        if columns[i].type == "longdouble":
                            column_array = _longdouble_from_bytes(byte_array, endianness)
                        else:
                            column_array = np.frombuffer(byte_array, dtype=columns_type[i], count=page_size)
                        if type_len == 1:
                            # Decode uint8 to <U1 to object
                            column_array = np.char.decode(column_array.view("S1"), "ascii").astype(object)
                        # l.debug(f'>>COL {i} {file.tell()} | {columns_type[i]} | {columns_size[i]} | {s} | {columns_data[i][row]} | {b_array}')
                if TRACE:
                    logger.debug(f">COL {i} END | {file.tell()=}")

                # Assign data
                if flag:
                    sdds.columns[i].data.append(column_array)
                    sdds.columns[i]._page_numbers.append(page_idx)

            if not page_skip:
                page_stored_idx += 1
        elif columns_all_numeric and sdds._meta_fixed_rowcount:
            for i in range(n_columns):
                if columns_mask[i]:
                    # struct yields 1-byte bytes for characters; collect as S1 and decode once at the end
                    columns_data.append(np.empty(page_size, dtype=columns_cell_dtype[i]))
            st = struct.Struct(combined_struct)
            page_size_actual = page_size  # reduced if a fixed-rowcount file ends early
            for row in range(page_size):
                byte_array = file.read(combined_size)
                if len(byte_array) < combined_size:
                    if sdds._meta_fixed_rowcount:
                        logger.info(f"Encountered fixed rowcount file end at row {row} of {page_size}")
                        page_size_actual = row
                        fixed_rowcount_eof = True
                        if len(byte_array) > 0:
                            logger.debug(f"Have leftover bytes {byte_array!r} in fixed rowcount mode, ignoring")
                        break
                    else:
                        raise ValueError(f"Unexpected EOF at row {row}")
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
                        else:
                            arr = columns_data[idx_active]
                        c.data.append(_finish_column(i, arr))
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1
        elif (
            columns_all_numeric
            and not sdds._meta_fixed_rowcount
            and sdds._source_file_size is not None
            and sdds._source_file_size > 500e6
        ):
            for i in range(n_columns):
                if columns_mask[i]:
                    columns_data.append(np.empty(page_size, dtype=columns_cell_dtype[i]))
            st = struct.Struct(combined_struct)
            byte_array = file.read(combined_size * page_size)
            if len(byte_array) < combined_size * page_size:
                raise ValueError(f"Unexpected EOF - got {len(byte_array)} bytes, wanted {combined_size * page_size}")
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
                        c.data.append(_finish_column(i, columns_data[idx_active]))
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1
        elif columns_all_numeric and not sdds._meta_fixed_rowcount:
            if (combined_size * page_size) % combined_dtype.itemsize != 0:
                raise ValueError(
                    f"Type length mismatch: {combined_size=} {page_size=} {combined_size*page_size=}"
                    f" {combined_dtype.itemsize=} {(combined_size * page_size) % combined_dtype.itemsize=}"
                )

            if page_skip or skip_all_columns:
                # Fast skip
                file.seek(combined_size * page_size, os.SEEK_CUR)
                if not page_skip:
                    page_stored_idx += 1
            else:
                byte_array = file.read(combined_size * page_size)
                if len(byte_array) != combined_size * page_size:
                    raise ValueError(
                        f"Unexpected EOF - got {len(byte_array)} bytes, wanted {combined_size * page_size}"
                    )

                array = np.frombuffer(byte_array, combined_dtype)
                idx_active = 0
                for i, c in enumerate(columns):
                    if columns_mask[i]:
                        if columns_len[i] == 1 or c.type == "longdouble":
                            c.data.append(_finish_column(i, array[str(i)]))
                        else:
                            # For now, make a copy to be safe
                            c.data.append(array[str(i)].copy())
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1
        else:
            for i in range(n_columns):
                if columns_mask[i]:
                    # character cells are decoded to str on assignment, so they take the store (object) dtype
                    dt = columns_store_type[i] if columns_len[i] == 1 else columns_cell_dtype[i]
                    columns_data.append(np.empty(page_size, dtype=dt))
            page_size_actual = page_size  # reduced if a fixed-rowcount file ends early
            store = not page_skip
            for row in range(page_size):
                if TRACE:
                    logger.debug(f">COL ROW {row} | {file.tell()=}")
                for seg_struct, seg_size, seg_numeric, has_string, str_dest in row_plan:
                    byte_array = file.read(seg_size)
                    if len(byte_array) < seg_size:
                        if sdds._meta_fixed_rowcount:
                            logger.info(f"Encountered fixed rowcount file end at row {row} of {page_size}")
                            page_size_actual = row
                            fixed_rowcount_eof = True
                            if len(byte_array) > 0:
                                raise ValueError(f"Weird leftover {byte_array}")
                            break
                        else:
                            raise ValueError(f"Unexpected EOF at row {row}")
                    values = seg_struct.unpack(byte_array)
                    if store:
                        for pos, d, is_char in seg_numeric:
                            # struct 'c' yields a 1-byte bytes object; store a plain str like the other paths
                            columns_data[d][row] = values[pos].decode("ascii") if is_char else values[pos]
                    if has_string:
                        string_len_actual = values[-1]
                        if string_len_actual < 0:
                            raise ValueError(
                                f"Column string length ({string_len_actual}) is negative - file is likely corrupt"
                            )
                        assert string_len_actual <= 1000000  # sanity check
                        if string_len_actual == 0:
                            if store and str_dest is not None:
                                columns_data[str_dest][row] = ""
                        else:
                            byte_array = file.read(string_len_actual)
                            if store and str_dest is not None:
                                columns_data[str_dest][row] = byte_array.decode("ascii")
                if TRACE:
                    logger.debug(f">COL END {row} | {file.tell()=}")
                if fixed_rowcount_eof:
                    break

            # Assign data to the columns
            if not page_skip:
                idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        if sdds._meta_fixed_rowcount and page_size_actual < page_size:
                            arr = columns_data[idx_active][:page_size_actual]
                        else:
                            arr = columns_data[idx_active]
                        if c.type == "longdouble":
                            arr = _finish_column(i, arr)
                        c.data.append(arr)
                        c._page_numbers.append(page_idx)
                        idx_active += 1
                page_stored_idx += 1

            if TRACE:
                logger.debug(f"Page {page_idx} data copy finished")

        page_idx += 1

        if fixed_rowcount_eof:
            break

        if file_size is not None:
            pos = file.tell()
            if pos < file_size:
                # More data exists
                if pages_mask is not None and page_idx == len(pages_mask):
                    logger.debug(f"Pages mask {pages_mask} is too short - expect {file_size - pos} more bytes")
                    break
            elif pos > file_size:
                raise Exception(f"{pos=} is beyond file size {file_size=}???")
            else:
                # End of file
                break
        else:
            next_byte = file.peek(1)
            if len(next_byte) > 0:
                # More data exists
                if pages_mask is not None and page_idx == len(pages_mask):
                    logger.debug(f"Pages mask {pages_mask} is too short - have at least {len(next_byte)} more bytes")
                    break
            else:
                # End of file
                break

    if flip_bytes:
        # parameters should already be in native format
        for i, el in enumerate(sdds.arrays):
            if arrays_mask[i] and el.type not in ["string", "character"]:
                for j in range(len(el.data)):
                    el.data[j] = el.data[j].astype(arrays_store_type[i], copy=False)  # .newbyteorder().byteswap()

        for i, el in enumerate(sdds.columns):
            if columns_mask[i] and el.type not in ["string", "character"]:
                for j in range(len(el.data)):
                    # If struct parser was used, data is already native, otherwise need to flip
                    el.data[j] = el.data[j].astype(columns_store_type[i], copy=False)
        logger.info(f"Data converted to native {sys.byteorder}-endian format, disable for max performance")
    sdds.n_pages = page_stored_idx


def _read_pages_ascii_mixed_lines(
    file: IO[bytes],
    sdds: SDDSFile,
    arrays_mask: List[bool],
    columns_mask: List[bool],
    pages_mask: List[bool],
) -> None:
    """Line by line numeric data parser for lines_per_row == 1"""

    all_parameters = sdds.parameters
    parameters = [p for p in all_parameters if p.fixed_value is None]
    parameters_type = [_NUMPY_DTYPES[el.type] for el in parameters]
    n_parameters = len(parameters)
    if n_parameters > 0:
        logger.debug("Parameter types: %s", parameters_type)

    arrays = sdds.arrays
    arrays_type = [_NUMPY_DTYPES[el.type] for el in arrays]
    n_arrays = len(arrays)
    if n_arrays > 0:
        logger.debug("Array types: %s", arrays_type)

    columns = sdds.columns
    n_columns = len(columns)
    columns_type = [_NUMPY_DTYPES[el.type] for el in columns]
    columns_store_type = [_NUMPY_DTYPE_FINAL[el.type] for el in columns]
    pd_column_dict = {
        i: columns_store_type[i] if columns_store_type[i] is not object else str for i in range(len(columns_type))
    }
    assert object in columns_type
    # Per-cell converters: Python float/int assign straight into the preallocated arrays, a few times cheaper
    # than np.fromstring per value and without creating a numpy scalar each time
    columns_conv = [None if t is object else _ascii_cell_converter(t) for t in columns_type]
    # pandas would parse longdouble cells as binary64 (and QUOTE_NONNUMERIC converts them before any dtype
    # applies), so pages with such columns always take the per-cell parser
    columns_have_longdouble = any(c.type == "longdouble" for c in columns)
    struct_type = None
    if n_columns > 0:
        logger.debug(f"Column types: {columns_type}")
        logger.debug(f"struct_type: {struct_type}")

    page_idx = 0
    page_stored_idx = 0
    # Flag for eof since can't break out of two loops
    while True:
        if pages_mask is not None:
            if page_idx >= len(pages_mask):
                logger.debug(f"Reached last page {page_idx} in mask, have at least 1 more remaining but exiting early")
                break
            else:
                page_skip = not pages_mask[page_idx]
        else:
            page_skip = False

        if page_skip:
            logger.debug(f">>PG | pos {file.tell()} | skipping page {page_idx}")
        else:
            logger.debug(">>PG | pos %d | reading page %d", file.tell(), page_idx)

        # Read parameters
        par_idx = 0
        while par_idx < n_parameters:
            b_array = __get_next_line(file, strip=True)
            if b_array is None:
                raise Exception(
                    f"Unexpected EOF during parameter parsing at page {page_idx} | {page_idx=} {par_idx=} | pos {file.tell()}"
                )
            if not page_skip:
                if parameters_type[par_idx] is object:
                    value = _decode_ascii_text_parameter(b_array, parameters[par_idx].type)
                    if TRACE:
                        logger.debug(
                            f">>PARS | p {file.tell()} | {par_idx=} | {parameters_type[par_idx]} "
                            f"| {b_array!r} | {value} | {parameters[par_idx].type}"
                        )
                else:
                    # Primitive types
                    value = np.fromstring(b_array, dtype=parameters_type[par_idx], sep=" ", count=1)[0]
                    if TRACE:
                        logger.debug(
                            f">>PARV | p {file.tell()} | {par_idx=} | {parameters_type[par_idx]} "
                            f"| {b_array!r} | {value} | {parameters[par_idx].type}"
                        )
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
                raise Exception(f">>ARRS | pos {file.tell()} | unexpected EOF at page {page_idx}")

            dimensions = np.fromstring(b_array, dtype=int, sep=" ")
            n_elements = np.prod(dimensions)
            if len(dimensions) != a.dimensions:
                raise ValueError(f">>Array {a.name} dimensions {b_array} did not match expected count {a.dimensions}")
            logger.debug(f">>Array {a.name} has dimensions {dimensions}, total of {n_elements}")

            # Start reading array
            n_lines_read = 0
            n_elements_read = 0
            line_values = []
            if arrays_type[array_idx] is object:
                # Strings need special treatment
                while True:
                    b_array = __get_next_line(file).strip()
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f">>ARRV | {file.tell()} | unexpected EOF at page {page_idx}")
                    values = split_sdds(b_array, posix=True)
                    logger.debug(
                        f">>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {b_array!r} | {values} | {n_elements=} | {n_lines_read=}"
                    )
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f"Too many elements read during array parsing: {n_elements_read} (need {n_elements})"
                        )
            else:
                # Primitive types
                while True:
                    b_array = __get_next_line(file)
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f">>ARRV | {file.tell()} | unexpected EOF at page {page_idx}")
                    values = np.fromstring(b_array, dtype=mapped_t, sep=" ", count=-1)
                    if TRACE:
                        logger.debug(
                            f">>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {b_array!r} | {values} | {n_elements=} | {n_lines_read=}"
                        )
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f"Too many elements read during array parsing: {n_elements_read} (need {n_elements})"
                        )

            if arrays_mask[array_idx] and not page_skip:
                values = np.concatenate(line_values)
                # Arrays are initialized in C order by default, matching SDDS
                if mapped_t is str:
                    data_array = np.empty(dimensions, dtype=object)
                else:
                    data_array = np.empty(dimensions, dtype=mapped_t)
                data_array[:] = values.reshape(dimensions)
                arrays[array_idx].data.append(data_array)
            array_idx += 1

        if sdds.data.no_row_counts:
            logger.debug(f">>C ({file.tell()}) | starting no_row_count parsing")
            page_size = None
        else:
            # Read column page size
            b_array = __get_next_line(file)
            if b_array is None:
                raise Exception(f">>COLS | {file.tell()} | unexpected EOF at page {page_idx}")

            page_size = int(b_array)
            assert 0 <= page_size <= 1e7
            logger.debug(f">>C ({file.tell()}) | page {page_idx} declared size {page_size}")

        # line = file.readline().decode('ascii')
        # list instead of generator to hopefully preallocate space
        if _ASCII_TEXT_PARSE_METHOD == "read_table" and not columns_have_longdouble:
            # Because read_table will consume too much if allowed to touch file, have to copy out a single page
            # TODO: see if maybe wrapping file will have higher perf
            lines = [file.readline().decode("ascii") for i in range(page_size)]
            buf = io.StringIO("\n".join(l for l in lines if not l.startswith("!")))
            # buf = io.StringIO('\n'.join(lines))
            # lines = [file.readline() for i in range(page_size)]
            # buf = io.BytesIO(b''.join(lines))
            opts = {
                "sep": r"\s+",
                "comment": "!",
                "header": None,
                "escapechar": "\\",
                "nrows": page_size,
                "skip_blank_lines": True,
                "skipinitialspace": True,
                "doublequote": False,
                "dtype": pd_column_dict,
                "engine": "c",
                "float_precision": "round_trip",  # the default C float parser is off by 1 ulp for some values
                "low_memory": False,
                "na_filter": False,
                "na_values": None,
                "quotechar": '"',
                "quoting": csv.QUOTE_NONNUMERIC,
                "keep_default_na": False,
            }
            # iowrap = io.TextIOWrapper(file, encoding='ascii')
            # df = pd.read_table(iowrap, **opts)
            # iowrap.detach()
            df = pd.read_table(buf, encoding="ascii", **opts)
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
        elif page_size is not None:
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
                        raise ValueError(f"Unexpected empty string at position {file.tell()}")

                    col_idx_active = 0
                    values = _split_ascii_row(line)
                    if TRACE:
                        logger.debug(f">COL ROW {row} | {len(values)}: {values=}")
                    for col_idx, c in enumerate(sdds.columns):
                        if columns_mask[col_idx]:
                            conv = columns_conv[col_idx]
                            if conv is None:
                                value = values[col_idx]
                            else:
                                value = conv(values[col_idx])
                            columns_data[col_idx_active][row] = value
                            if TRACE:
                                logger.debug(f">>CR {row=} | {c.name}:{value}")
                            col_idx_active += 1

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
        elif _ASCII_TEXT_PARSE_METHOD == "shlex" and page_size is None:
            # no_row_counts parsing
            columns_list_data = []
            if not page_skip:
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        columns_list_data.append([])
            row_cnt = 0
            while True:
                line = __get_next_line(
                    file,
                    accept_meta_commands=False,
                    strip=False,
                    cut_midline_comments=False,
                )
                if line is None or line == "\n" or line == "\r\n":
                    logger.debug(f">>C ({file.tell()}) | End of no_row_counts page at row {row_cnt}")
                    break
                if not page_skip:
                    line_len = len(line)
                    if line_len == 0:
                        raise ValueError(f"Unexpected empty string at position {file.tell()}")
                    if row_cnt > 1e7:
                        raise SDDSReadError(f"Read more than {row_cnt} rows - something is wrong")
                    col_idx_active = 0
                    values = _split_ascii_row(line.strip())
                    if TRACE:
                        logger.debug(f">C ({file.tell()}) | {len(values)}: {values=}")
                    for col_idx, c in enumerate(sdds.columns):
                        if columns_mask[col_idx]:
                            conv = columns_conv[col_idx]
                            if conv is None:
                                value = values[col_idx]
                            else:
                                value = conv(values[col_idx])
                            columns_list_data[col_idx_active].append(value)
                            if TRACE:
                                logger.debug(f">>C ({file.tell()}) | {c.name}:{value}")
                            col_idx_active += 1
                    row_cnt += 1

            # Assign data to the columns
            if not page_skip:
                # With no columns selected there is nothing to store, but the page still counts
                column_lengths = [len(x) for x in columns_list_data]
                assert len(set(column_lengths)) <= 1, f"Column mismatch {column_lengths}"
                col_idx_active = 0
                for i, c in enumerate(sdds.columns):
                    if columns_mask[i]:
                        c.data.append(
                            np.array(
                                columns_list_data[col_idx_active],
                                dtype=columns_store_type[i],
                            )
                        )
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
            raise Exception(f"Unrecognized parse method: {_ASCII_TEXT_PARSE_METHOD}")

        while True:
            # Look for next important character (this is rough heuristic)
            next_byte = file.peek(1)
            if len(next_byte) > 0:
                next_char = next_byte[:1].decode("ascii")
                # print(repr(next_char))
                if next_char == "\n":
                    file.read(1)
                    continue
                elif next_char == "\r":
                    file.read(1)
                    next_byte = file.peek(1)
                    if len(next_byte) > 0 and next_byte[:1].decode("ascii") == "\n":
                        file.read(1)
                    else:
                        raise SDDSReadError(f"Unexpected \\r without \\n at {file.tell()}")
                else:
                    logger.debug(f"Found character {next_char!r} at {file.tell()}, continuing to next page")
                    break
            else:
                break
        if len(next_byte) > 0:
            # More data exists
            if pages_mask is not None and page_idx == len(pages_mask):
                logger.debug(f"Mask {pages_mask} ended but have at least {len(next_byte)} extra bytes - stopping")
                break
        else:
            # End of file
            break
    sdds.n_pages = page_stored_idx


def _read_pages_ascii_numeric_lines(
    file: IO[bytes],
    sdds: SDDSFile,
    arrays_mask: List[bool],
    columns_mask: List[bool],
    pages_mask: List[bool],
) -> None:
    """Line by line numeric data parser for lines_per_row == 1"""

    all_parameters = sdds.parameters
    parameters = [p for p in all_parameters if p.fixed_value is None]
    n_params_unfixed = sum(p.fixed_value is None for p in sdds.parameters)
    parameter_types = [_NUMPY_DTYPES[el.type] for el in parameters]
    logger.debug(f"Parameter types: {parameter_types}")

    arrays = sdds.arrays
    arrays_type = [_NUMPY_DTYPES[el.type] for el in arrays]

    columns = sdds.columns
    n_columns = len(columns)
    if n_columns > 0:
        columns_type = [_NUMPY_DTYPES[el.type] for el in columns]
        # columns_store_type = [_NUMPY_DTYPE_FINAL[el.type] for el in columns]
        assert object not in columns_type
        struct_type = np.dtype(", ".join(columns_type))

        logger.debug(f"Column types: {columns_type}")
        logger.debug(f"Column struct_type: {struct_type}")
    else:
        columns_type = []
        # columns_store_type = []

    columns_conv = [_ascii_cell_converter(t) for t in columns_type]

    def _parse_small_page(lines, pg_idx):
        # Plain split-and-convert into preallocated arrays; only for small pages where pandas setup dominates
        active = [
            (i, np.empty(len(lines), dtype=columns_type[i]), columns_conv[i])
            for i in range(n_columns)
            if columns_mask[i]
        ]
        for r, l in enumerate(lines):
            if "!" in l:
                l = l[: l.index("!")]
            v = l.split()
            for i, arr, conv in active:
                arr[r] = conv(v[i])
        for i, arr, _ in active:
            columns[i].data.append(arr)
            columns[i]._page_numbers.append(pg_idx)

    def _append_empty_columns(pg_idx):
        # Zero-row page - store empty arrays of the right dtype for every active column
        for ci, col in enumerate(columns):
            if columns_mask[ci]:
                col.data.append(np.empty(0, dtype=columns_type[ci]))
                col._page_numbers.append(pg_idx)

    page_idx = 0
    page_stored_idx = 0

    b_array = __get_next_line(file)
    if b_array is None:
        # Empty file
        sdds.n_pages = 0
        return
    elif n_params_unfixed == 0 and len(arrays) == 0 and n_columns == 0:
        # Nothing is consumed per page, so ASCII pages are not delimited - treat any content as a single page
        # (without this, the page loop below would spin forever on the unconsumed line)
        sdds.n_pages = 1 if pages_mask is None or pages_mask[0] else 0
        return
    else:
        pushback_line_buf.appendleft(b_array)

    # Flag for eof since can't break out of two loops
    while True:
        if pages_mask is not None:
            if page_idx >= len(pages_mask):
                logger.debug(f"Reached last page {page_idx} in mask, have at least 1 more remaining but exiting early")
                break
            else:
                page_skip = not pages_mask[page_idx]
        else:
            page_skip = False

        if page_skip:
            logger.debug(f">>PG | pos {file.tell()} | skipping page {page_idx}")
        else:
            logger.debug(">>PG | pos %d | reading page %d", file.tell(), page_idx)

        # Read parameters
        parameter_data = []
        par_idx = 0
        par_line_num = 0
        if n_params_unfixed > 0:
            # TODO: mixed fixed/unfixed parameters
            while par_idx < len(parameter_types):
                b_array = __get_next_line(file, accept_meta_commands=False)
                if b_array is None:
                    raise Exception(f">>PARS | pos {file.tell()} | unexpected EOF at page {page_idx}")
                par_line_num += 1

                if par_line_num > 10000:
                    raise Exception("Still parsing parameters after 10000 lines - something is wrong")

                if parameter_types[par_idx] is object:
                    value = _decode_ascii_text_parameter(b_array, parameters[par_idx].type)
                    if TRACE:
                        logger.debug(
                            f">>PARS | pos {file.tell()} | {par_idx=} | {parameter_types[par_idx]} | {b_array!r} | {value}"
                        )
                else:
                    # Primitive types
                    value = np.fromstring(b_array, dtype=parameter_types[par_idx], sep=" ", count=1)[0]
                    if TRACE:
                        logger.debug(
                            f">>PARV | pos {file.tell()} | {par_idx=} | {parameter_types[par_idx]} | {b_array!r} | {value}"
                        )
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
            b_array = __get_next_line(file, accept_meta_commands=False)
            if b_array is None:
                raise Exception(f">>ARRS | pos {file.tell()} | unexpected EOF at page {page_idx}")

            dimensions = np.fromstring(b_array, dtype=int, sep=" ", count=-1)
            n_elements = np.prod(dimensions)
            if len(dimensions) != a.dimensions:
                raise ValueError(f">>Array {a.name} dimensions {b_array} did not match expected count {a.dimensions}")
            logger.debug(f">>Array {a.name} has dimensions {dimensions}, total of {n_elements}")

            # Start reading array
            n_lines_read = 0
            n_elements_read = 0
            line_values = []
            if arrays_type[array_idx] is object:
                # Strings need special treatment
                while True:
                    b_array = __get_next_line(file).strip()
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f">>ARRV | {file.tell()} | unexpected EOF at page {page_idx}")
                    values = split_sdds(b_array, posix=True)
                    logger.debug(
                        f">>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {b_array!r} | {values} | {n_elements=} | {n_lines_read=}"
                    )
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f"Too many elements read during array parsing: {n_elements_read} (need {n_elements})"
                        )
            else:
                # Primitive types
                while True:
                    b_array = __get_next_line(file)
                    n_lines_read += 1
                    if b_array is None:
                        raise Exception(f">>ARRV | {file.tell()} | unexpected EOF at page {page_idx}")
                    values = np.fromstring(b_array, dtype=mapped_t, sep=" ", count=-1)
                    if TRACE:
                        logger.debug(
                            f">>ARRV | {file.tell()} | {array_idx=} | {mapped_t} | {b_array!r} | {values} | {n_elements=} | {n_lines_read=}"
                        )
                    n_elements_read += len(values)
                    line_values.append(values)
                    if n_elements_read < n_elements:
                        continue
                    elif n_elements_read == n_elements:
                        # Done
                        break
                    else:
                        raise Exception(
                            f"Too many elements read during array parsing: {n_elements_read} (need {n_elements})"
                        )

            if arrays_mask[array_idx] and not page_skip:
                values = np.concatenate(line_values)
                # Arrays are initialized in C order by default, matching SDDS
                if mapped_t is str:
                    data_array = np.empty(dimensions, dtype=object)
                else:
                    data_array = np.empty(dimensions, dtype=mapped_t)
                data_array[:] = values.reshape(dimensions)
                arrays[array_idx].data.append(data_array)
            array_idx += 1

        if n_columns == 0:
            pass
        else:
            if sdds.data.no_row_counts:
                logger.debug(f">>C ({file.tell()}) | starting no_row_count numerical ascii")
                page_size = None
            else:
                # Read column page size
                b_array = __get_next_line(file)
                if b_array is None:
                    raise Exception(f">>C {file.tell()} | unexpected EOF at page {page_idx}")

                page_size = int(b_array)
                assert 0 <= page_size <= 1e7
                logger.debug(f">>C {file.tell()} | page size: {page_size}")

            # line = file.readline().decode('ascii')
            # list instead of generator to hopefully preallocate space
            if _ASCII_NUMERIC_PARSE_METHOD == "loadtxt" and page_size is not None:
                gen = (
                    __get_next_line(file, accept_meta_commands=False, strip=True, replace_tabs=True)
                    for _ in range(page_size)
                )
                if not page_skip:
                    data = np.loadtxt(
                        gen,
                        dtype=struct_type,
                        unpack=True,
                        usecols=np.where(columns_mask)[0],
                        comments="!",
                    )
                    print(len(data), len(columns), len(columns_mask))
                    col_idx_active = 0
                    for col_idx, c in enumerate(columns):
                        if columns_mask[col_idx]:
                            c.data.append(data[col_idx_active])
                            c._page_numbers.append(page_idx)
                            col_idx_active += 1
                    page_stored_idx += 1
            elif _ASCII_NUMERIC_PARSE_METHOD == "loadtxt" and page_size is None:

                def gen():
                    l = __get_next_line(file, accept_meta_commands=False, strip=False, replace_tabs=True)
                    if l is None or l == "\n" or l == "\r\n":
                        return
                    yield l

                if not page_skip:
                    data = np.loadtxt(
                        gen(),
                        dtype=struct_type,
                        unpack=True,
                        usecols=np.where(columns_mask)[0],
                        comments="!",
                    )
                    logger.debug(
                        f">>C {file.tell()} | Parse no_row_count page of length {len(data)}"
                        f" with {len(columns)=} {len(columns_mask)=}"
                    )
                    col_idx_active = 0
                    for col_idx, c in enumerate(columns):
                        if columns_mask[col_idx]:
                            c.data.append(data[col_idx_active])
                            c._page_numbers.append(page_idx)
                            col_idx_active += 1
                    page_stored_idx += 1
            elif _ASCII_NUMERIC_PARSE_METHOD == "read_table" and page_size == 0:
                # pandas raises EmptyDataError on an empty buffer, so build the empty columns directly
                if not page_skip:
                    _append_empty_columns(page_idx)
            elif _ASCII_NUMERIC_PARSE_METHOD == "read_table" and page_size is not None:
                # pandas parses floats as binary64, so longdouble columns are read as text and converted after
                pd_column_dict = {
                    i: (str if columns[i].type == "longdouble" else columns_type[i]) for i in range(len(columns_type))
                }
                # Collect exactly page_size data lines - comment lines do not count towards the row total
                lines = []
                while len(lines) < page_size:
                    l = file.readline().decode("ascii")
                    if not l:
                        raise SDDSReadError(f"Unexpected EOF in page {page_idx} after {len(lines)} of {page_size} rows")
                    if "!" in l and l.lstrip().startswith("!"):
                        continue
                    lines.append(l)
                if page_size <= _ASCII_SMALL_PAGE_ROWS:
                    if not page_skip:
                        _parse_small_page(lines, page_idx)
                else:
                    buf = io.StringIO("\n".join(lines))
                    opts = {
                        "sep": r"\s+",
                        "comment": "!",
                        "header": None,
                        "escapechar": "\\",
                        "nrows": page_size,
                        "skip_blank_lines": True,
                        "skipinitialspace": True,
                        "doublequote": False,
                        "dtype": pd_column_dict,
                        "engine": "c",
                        "float_precision": "round_trip",  # the default C float parser is off by 1 ulp for some values
                        "low_memory": False,
                        "na_filter": False,
                        "na_values": None,
                        "keep_default_na": False,
                    }
                    df = pd.read_table(buf, encoding="ascii", **opts)
                    # Assign data to the columns (pandas parsed every column, so index by file column)
                    if not page_skip:
                        for i, c in enumerate(sdds.columns):
                            if columns_mask[i]:
                                values = df.iloc[:, i].values
                                if c.type == "longdouble":
                                    values = values.astype(_NUMPY_DTYPE_FINAL["longdouble"])
                                c.data.append(values)
                                c._page_numbers.append(page_idx)
            elif _ASCII_NUMERIC_PARSE_METHOD == "read_table" and page_size is None:
                # no_row_count mode
                # TODO: exponential growth buffer
                # pandas parses floats as binary64, so longdouble columns are read as text and converted after
                pd_column_dict = {
                    i: (str if columns[i].type == "longdouble" else columns_type[i]) for i in range(len(columns_type))
                }
                cnt = 0
                line_cnt = 0

                def gen():
                    nonlocal cnt, line_cnt
                    while True:
                        l = __get_next_line(
                            file,
                            accept_meta_commands=False,
                            strip=False,
                            replace_tabs=True,
                        )
                        if l is None or l == "\n" or l == "\r\n":
                            return
                        l.strip()
                        cnt += len(l)
                        line_cnt += 1
                        yield l

                buf = io.StringIO("\n".join(line for line in gen()))
                logger.debug(f">>C {file.tell()} | Feeding buffer {cnt=} {line_cnt=} to parse_table")
                if line_cnt == 0:
                    # Page ended immediately - pandas cannot parse an empty buffer
                    # (fall through to the common page bookkeeping and EOF check below)
                    if not page_skip:
                        _append_empty_columns(page_idx)
                else:
                    opts = {
                        "sep": r"\s+",
                        "comment": "!",
                        "header": None,
                        "escapechar": "\\",
                        "nrows": line_cnt,
                        "skip_blank_lines": True,
                        "skipinitialspace": True,
                        "doublequote": False,
                        "dtype": pd_column_dict,
                        "engine": "c",
                        "float_precision": "round_trip",  # the default C float parser is off by 1 ulp for some values
                        "low_memory": False,
                        "na_filter": False,
                        "na_values": None,
                        "keep_default_na": False,
                    }
                    df = pd.read_table(buf, encoding="ascii", **opts)
                    # Assign data to the columns
                    if not page_skip:
                        for i, c in enumerate(sdds.columns):
                            if columns_mask[i]:
                                values = df.iloc[:, i].values
                                if c.type == "longdouble":
                                    values = values.astype(_NUMPY_DTYPE_FINAL["longdouble"])
                                c.data.append(values)
                                c._page_numbers.append(page_idx)
        page_idx += 1
        if not page_skip:
            page_stored_idx += 1

        while True:
            # Look for next important character (this is rough heuristic)
            next_byte = file.peek(1)
            if len(next_byte) > 0:
                next_char = next_byte[:1].decode("ascii")
                # print(repr(next_char))
                if next_char == "\n":
                    file.read(1)
                    continue
                else:
                    break
            else:
                break

        if len(next_byte) > 0:
            # More data exists
            if pages_mask is not None and page_idx == len(pages_mask):
                logger.debug(f"Mask {pages_mask} ended but have at least {len(next_byte)} extra bytes - stopping")
                break
        else:
            # End of file
            break
    sdds.n_pages = page_stored_idx
