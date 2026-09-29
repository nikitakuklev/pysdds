"""Public API regressions for parser boundaries and selective data conversion."""

import bz2
import gzip
import io
import lzma
import struct

import numpy as np
import pandas as pd
import pytest

import pysdds
from pysdds.util.errors import SDDSReadError


def _header(definitions, data="mode=ascii", meta=""):
    return ("SDDS1\n" + meta + definitions + f"&data {data}, &end\n").encode("ascii")


def _read(data, **kwargs):
    return pysdds.read(io.BytesIO(data), **kwargs)


@pytest.mark.parametrize("endian,prefix", [("big", ">"), ("little", "<")])
@pytest.mark.parametrize("rows", [0, 1])
def test_explicit_endianness_without_metadata(endian, prefix, rows):
    h = _header("&parameter name=p, type=double, &end\n&column name=x, type=long, &end\n", "mode=binary")
    page = struct.pack(prefix + "id", rows, 1.25) + struct.pack(prefix + "i" * rows, *([42] * rows))
    result = _read(h + page * 2, endianness=endian)
    assert result.endianness == endian
    assert result.par("p").data == [1.25, 1.25]
    assert [x.tolist() for x in result.col("x").data] == [[42] * rows] * 2


@pytest.mark.parametrize("declaration", ["meta", "data", "both"])
def test_endian_conflicts(declaration):
    meta = "!# little-endian\n" if declaration in ("meta", "both") else ""
    data = "mode=binary" + (", endian=little" if declaration in ("data", "both") else "")
    with pytest.raises(ValueError, match="does not match"):
        _read(_header("", data, meta), endianness="big", header_only=True)


@pytest.mark.parametrize("cmo", [0, 1])
@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("kind", ["column", "array", "parameter"])
def test_truncated_binary_string_payload(kind, selected, cmo):
    h = _header(f"&{kind} name=s, type=string, &end\n", f"mode=binary, column_major_order={cmo}")
    payload = struct.pack("<i", 1 if kind == "column" else 0)
    if kind == "array":
        payload += struct.pack("<i", 1)
    payload += struct.pack("<i", 5) + b"ab"
    kwargs = {} if selected or kind == "parameter" else {"cols" if kind == "column" else "arrays": []}
    with pytest.raises(SDDSReadError, match="EOF"):
        _read(h + payload, **kwargs)


@pytest.mark.parametrize("tail", [b"\x02", struct.pack("<ii", 2, 5) + b"ab", struct.pack("<ii", 2, 1) + b"b\x01"])
def test_fixed_rowcount_discards_entire_incomplete_row(tail):
    h = _header(
        "&column name=x, type=long, &end\n&column name=s, type=string, &end\n&column name=y, type=long, &end\n",
        "mode=binary",
        "!# fixed-rowcount\n",
    )
    complete = struct.pack("<ii", 1, 1) + b"a" + struct.pack("<i", 10)
    result = _read(h + struct.pack("<i", 10) + complete + tail)
    result.validate_data()
    assert result.col("x").data[0].tolist() == [1]
    assert result.col("s").data[0].tolist() == ["a"]
    assert result.col("y").data[0].tolist() == [10]


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("atype", ["double", "string", "character"])
@pytest.mark.parametrize("shape", [(0,), (3, 0)])
@pytest.mark.parametrize("legacy_blank", [False, True])
def test_empty_ascii_arrays_preserve_next_section(mixed, atype, shape, legacy_blank):
    column_type, cell = ("string", "hello") if mixed else ("long", "42")
    h = _header(
        f"&array name=a, type={atype}, dimensions={len(shape)}, &end\n"
        "&array name=b, type=long, &end\n"
        f"&column name=x, type={column_type}, &end\n"
    )
    page = " ".join(map(str, shape)) + " ! empty array\n" + ("\n" if legacy_blank else "")
    page += f"2\n8 9\n1\n{cell}\n"
    result = _read(h + (page * 2).encode())
    assert result.n_pages == 2
    assert [a.shape for a in result.array("a").data] == [shape, shape]
    assert result.array("b").data[1].tolist() == [8, 9]
    buf = io.BytesIO()
    pysdds.write(result, buf, use_best_settings=False)
    reread = _read(buf.getvalue())
    assert reread.array("a").data[1].shape == shape


@pytest.mark.parametrize("count", [0, 2])
def test_additional_header_lines_count_noncomments_only(count):
    h = _header("&column name=x, type=long, &end\n", f"mode=ascii, additional_header_lines={count}")
    extra = b"! comment\nignored\n! another comment\n\n" if count else b""
    result = _read(h + extra + b"1\n42\n1\n43\n")
    assert result.data.additional_header_lines == count
    assert [a.tolist() for a in result.col("x").data] == [[42], [43]]


@pytest.mark.parametrize("count", ["-1", "x", "1.5"])
def test_invalid_additional_header_lines(count):
    with pytest.raises(ValueError):
        _read(_header("", f"mode=ascii, additional_header_lines={count}"))


def test_header_comments_quotes_escapes_and_multiline_values():
    data = (
        b"SDDS1\n"
        b'&description text="first ! &end\nsecond \\"quoted\\"", contents="ok", &end ! comment " ! again\n'
        b'&column name=x, ! comment with " unmatched quote\n'
        b' type=long, description="literal ! and \\" quote", &end ! tail\n'
        b"&data mode=ascii, &end! final header comment\n1\n42\n"
    )
    result = _read(data)
    assert result.description.text == 'first ! &end\nsecond "quoted"'
    assert result.col("x").data[0].tolist() == [42]


@pytest.mark.parametrize("selection", [lambda: iter([0]), lambda: np.array([0], dtype=np.int64)])
def test_iterator_and_numpy_integer_selections(selection):
    h = _header(
        "&array name=a, type=long, &end\n&array name=b, type=long, &end\n"
        "&column name=x, type=long, &end\n&column name=y, type=long, &end\n"
    )
    page = b"1\n8\n1\n9\n1\n42 43\n"
    result = _read(h + page * 2, cols=selection(), arrays=selection(), pages=selection())
    assert result.n_pages == 1
    assert result.col("x").data[0].tolist() == [42]
    assert result.array("a").data[0].tolist() == [8]
    assert not result.col("y")._enabled and not result.array("b")._enabled


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("nrc", [0, 1])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_trailing_comments_preserve_blank_parameters_and_page_delimiters(mixed, nrc, newline):
    ctype, row = ("string", b"hello\n") if mixed else ("long", b"42\n")
    h = _header(
        f"&parameter name=p, type=string, &end\n&column name=x, type={ctype}, &end\n",
        f"mode=ascii, no_row_counts={nrc}",
    )
    page = b"\n" + (b"" if nrc else b"1\n") + row
    payload = h + page + (b"\n" if nrc else b"") + b"! page\n" + page + b"! trailing\n  ! trailing again\n"
    result = _read(payload.replace(b"\n", newline))
    assert result.n_pages == 2
    assert result.par("p").data == ["", ""]
    assert len(result.col("x").data) == 2


@pytest.mark.parametrize("nrc", [0, 1])
@pytest.mark.parametrize("cols,pages,expected_calls", [(["z", "x"], None, 2), ([], None, 0), (["z"], [1], 1)])
def test_pandas_converts_only_selected_pages_and_columns(monkeypatch, nrc, cols, pages, expected_calls):
    h = _header("".join(f"&column name={c}, type=double, &end\n" for c in "xyz"), f"mode=ascii, no_row_counts={nrc}")
    page = (b"" if nrc else b"1001\n") + b"1 2 3\n" * 1001
    payload = h + page + (b"\n" if nrc else b"") + page
    original = pd.read_table
    calls = []

    def observe(*args, **kwargs):
        df = original(*args, **kwargs)
        calls.append((kwargs["usecols"], df.shape))
        return df

    monkeypatch.setattr(pd, "read_table", observe)
    result = _read(payload, cols=cols, pages=pages)
    assert result.n_pages == (1 if pages else 2)
    assert len(calls) == expected_calls
    expected_indices = [i for i, c in enumerate("xyz") if c in cols]
    assert all(indices == expected_indices and shape == (1001, len(cols)) for indices, shape in calls)
    for c in cols:
        assert result.col(c).data[0][0] == "xyz".index(c) + 1


@pytest.mark.parametrize(
    "suffix,compress", [("", lambda b: b), (".gz", gzip.compress), (".bz2", bz2.compress), (".xz", lzma.compress)]
)
@pytest.mark.parametrize("cmo", [0, 1])
def test_binary_skipping_checks_eof_and_preserves_pages(tmp_path, suffix, compress, cmo):
    h = _header(
        "&array name=a, type=double, &end\n&column name=x, type=double, &end\n",
        f"mode=binary, column_major_order={cmo}",
    )
    page = struct.pack("<ii", 20000, 20000) + bytes(20000 * 8) + np.arange(20000, dtype="<f8").tobytes()
    path = tmp_path / ("skip.sdds" + suffix)
    path.write_bytes(compress(h + page + page))
    result = pysdds.read(path, arrays=[], pages=[1])
    assert result.n_pages == 1
    assert result.col("x").data[0][-1] == 19999
    for truncated in [page[:100], page[:-1]]:
        path.write_bytes(compress(h + truncated))
        with pytest.raises(SDDSReadError, match="EOF"):
            pysdds.read(path, arrays=[], cols=[])


def test_excluded_binary_reads_are_bounded():
    class Observed(io.BufferedIOBase):
        # Unknown logical size/non-seekable: exercise the bounded-discard fallback.
        def __init__(self, data):
            self.buffer = io.BytesIO(data)
            self.sizes = []

        def tell(self):
            return self.buffer.tell()

        def readline(self, size=-1):
            return self.buffer.readline(size)

        def peek(self, size):
            return self.buffer.getvalue()[self.tell() : self.tell() + size]

        def read(self, size=-1):
            self.sizes.append(size)
            return self.buffer.read(size)

    h = _header(
        "&array name=a, type=double, &end\n&column name=x, type=double, &end\n", "mode=binary, column_major_order=1"
    )
    stream = Observed(h + struct.pack("<ii", 20000, 20000) + bytes(20000 * 16))
    result = pysdds.read(stream, arrays=[], cols=[])
    assert result.n_pages == 1
    assert max(stream.sizes) <= 65536


def test_no_row_counts_parameter_only_writer_separators():
    payload = _header("&parameter name=p, type=long, &end\n", "mode=ascii, no_row_counts=1")
    result = _read(payload + b"1\n\n! page number 1\n2\n! trailing comment\n")
    assert result.par("p").data == [1, 2]
    assert result.n_pages == 2


@pytest.mark.parametrize("values", [["a", "b"], ["", "b", "", ""], ["a", "", "b"]])
@pytest.mark.parametrize("nrc", [0, 1])
def test_parameter_only_string_writer_roundtrip(values, nrc):
    source = pysdds.SDDSFile.from_df([pd.DataFrame() for _ in values], parameter_dict={"p": values}, mode="ascii")
    source.data.nm["no_row_counts"] = nrc
    output = io.BytesIO()
    pysdds.write(source, output, use_best_settings=False)
    result = _read(output.getvalue())
    assert result.n_pages == len(values)
    assert result.par("p").data == values


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_legacy_parameter_only_string_separators_and_blank_values(newline):
    h = _header("&parameter name=p, type=string, &end\n", "mode=ascii, no_row_counts=1")
    body = b"! page number 0\na\n\n! page number 1\n\n\n! page number 2\nb\n\n! page number 3\n\n"
    result = _read((h + body).replace(b"\n", newline))
    assert result.par("p").data == ["a", "", "b", ""]
    assert result.n_pages == 4


def test_parameter_only_blank_strings_without_page_markers():
    h = _header("&parameter name=p, type=string, &end\n", "mode=ascii, no_row_counts=1")
    result = _read(h + b"a\n\nb\n\n\n")
    assert result.par("p").data == ["a", "", "b", "", ""]


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("atype", ["string", "double"])
@pytest.mark.parametrize("nrc", [0, 1])
def test_legacy_empty_array_payload_before_columns(mixed, atype, nrc):
    ctype, first, second = ("string", "hello", "bye") if mixed else ("long", "42", "43")
    h = _header(
        f"&array name=a, type={atype}, &end\n&column name=x, type={ctype}, &end\n",
        f"mode=ascii, no_row_counts={nrc}",
    )
    count = "" if nrc else "1\n"
    separator = "\n" if nrc else ""
    body = f"! page number 0\n0 ! 1-dimensional array a\n\n{count}{first}\n{separator}"
    body += f"! page number 1\n1 ! 1-dimensional array a\n8\n{count}{second}\n"
    result = _read(h + body.encode())
    assert result.n_pages == 2
    assert result.array("a").data[0].shape == (0,)
    assert [x.tolist() for x in result.col("x").data] == ([["hello"], ["bye"]] if mixed else [[42], [43]])


def test_legacy_empty_array_payload_before_next_string_parameter():
    h = _header("&parameter name=p, type=string, &end\n&array name=a, type=string, &end\n")
    body = b"! page number 0\na\n0 ! 1-dimensional array a\n\n! page number 1\n\n1 ! 1-dimensional array a\nu\n"
    result = _read(h + body)
    assert result.n_pages == 2
    assert result.par("p").data == ["a", ""]
    assert result.array("a").data[1].tolist() == ["u"]


@pytest.mark.parametrize("comment", ["", " ! 1-dimensional array a:", " ! 1-dimensional array a"])
@pytest.mark.parametrize("mixed", [False, True])
def test_empty_array_and_empty_no_row_count_table_keep_page_boundary(comment, mixed):
    ctype, value = ("string", "hello") if mixed else ("long", "42")
    h = _header(
        f"&array name=a, type=string, &end\n&column name=x, type={ctype}, &end\n",
        "mode=ascii, no_row_counts=1",
    )
    body = f"! page number 0\n0{comment}\n\n! page number 1\n1 ! 1-dimensional array a:\nu\n{value}\n"
    result = _read(h + body.encode())
    assert result.n_pages == 2
    assert result.col("x").data[0].size == 0
    assert result.col("x").data[1].tolist() == (["hello"] if mixed else [42])


def test_unmarked_empty_array_preserves_unmarked_empty_table():
    h = _header("&array name=a, type=string, &end\n&column name=x, type=long, &end\n", "mode=ascii, no_row_counts=1")
    result = _read(h + b"0\n\n1\nu\n42\n")
    assert result.n_pages == 2
    assert result.col("x").data[0].size == 0
    assert result.array("a").data[1].tolist() == ["u"]
    assert result.col("x").data[1].tolist() == [42]


@pytest.mark.parametrize("mixed", [False, True])
def test_empty_arrays_with_empty_and_nonempty_tables_writer_roundtrip(mixed):
    from pysdds.structures import Array

    frames = [pd.DataFrame({"x": np.array(values, dtype=float)}) for values in [[42], [], []]]
    if mixed:
        for frame in frames:
            frame["s"] = pd.array(["hello"] * len(frame), dtype="string")
    source = pysdds.SDDSFile.from_df(frames, mode="ascii")
    for column in source.columns:
        column.data[0], column.data[1] = column.data[1], column.data[0]
    source.data.nm["no_row_counts"] = 1
    array = Array({"name": "a", "type": "string"}, source)
    array.data = [np.empty(0, dtype=object) for _ in frames]
    source.arrays.append(array)
    source.n_arrays = 1
    output = io.BytesIO()
    pysdds.write(source, output, use_best_settings=False)
    current = output.getvalue()
    # Reconstruct the old writer's dimension comment and its extra empty-array payload line.
    legacy = current.replace(b"! 1-dimensional array a:\n", b"! 1-dimensional array a\n\n")
    for payload in [current, legacy]:
        result = _read(payload)
        assert result.n_pages == 3
        assert [a.shape for a in result.array("a").data] == [(0,)] * 3
        assert [c.tolist() for c in result.col("x").data] == [[], [42], []]


@pytest.mark.parametrize("ctype", ["string", "double"])
def test_terminal_metadata_selection_does_not_read_compressed_column_tail(ctype):
    class ObservedGzip(gzip.GzipFile):
        def __init__(self, data):
            super().__init__(fileobj=io.BytesIO(gzip.compress(data)), mode="rb")
            self.read_sizes = []

        def read(self, size=-1):
            self.read_sizes.append(size)
            return super().read(size)

    h = _header(f"&parameter name=p, type=long, &end\n&column name=x, type={ctype}, &end\n", "mode=binary")
    payload = struct.pack("<ii", 100000, 42) + bytes(800000)
    stream = ObservedGzip(h + payload)
    result = pysdds.read(stream, cols=[], pages=[0])
    assert result.n_pages == 1
    assert result.par("p").data == [42]
    assert stream.read_sizes == [4, 4]


@pytest.mark.parametrize("layout", ["numeric", "mixed", "arrays", "arrays_numeric"])
@pytest.mark.parametrize("nrc", [0, 1])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
@pytest.mark.parametrize("trailer", [b"\n", b"\n\n", b"\n! trailing comment\n\n"])
def test_trailing_blanks_after_string_parameter_pages(layout, nrc, newline, trailer):
    definitions = "&parameter name=p, type=string, &end\n"
    has_arrays = layout.startswith("arrays")
    has_columns = layout != "arrays"
    if has_arrays:
        definitions += "&array name=a, type=long, &end\n"
    if has_columns:
        definitions += "&column name=x, type=long, &end\n"
    if layout == "mixed":
        definitions += "&column name=s, type=string, &end\n"
    h = _header(definitions, f"mode=ascii, no_row_counts={nrc}")
    body = b""
    for page, parameter in enumerate([b"a", b""]):
        if page and nrc and has_columns:
            body += b"\n"
        body += f"! page number {page}\n".encode() + parameter + b"\n"
        if has_arrays:
            body += b"1\n9\n"
        if has_columns:
            if not nrc:
                body += b"1\n"
            body += str(42 + page).encode() + (b" hello" if layout == "mixed" else b"") + b"\n"
    result = _read((h + body + trailer).replace(b"\n", newline))
    assert result.n_pages == 2
    assert result.par("p").data == ["a", ""]
    if has_columns:
        assert [c.tolist() for c in result.col("x").data] == [[42], [43]]
    if has_arrays:
        assert [a.tolist() for a in result.array("a").data] == [[9], [9]]


@pytest.mark.parametrize("nrc", [0, 1])
def test_blank_parameter_lookahead_restores_comments_and_multiple_blank_values(nrc):
    h = _header(
        "&parameter name=p, type=string, &end\n&parameter name=q, type=string, &end\n&column name=x, type=long, &end\n",
        f"mode=ascii, no_row_counts={nrc}",
    )
    count = b"" if nrc else b"1\n"
    body = b"a\nb\n" + count + b"42\n" + (b"\n" if nrc else b"")
    body += b"! page number 1\n\n! between parameters\n\n! before table\n" + count + b"43\n"
    result = _read(h + body)
    assert result.n_pages == 2
    assert result.par("p").data == ["a", ""]
    assert result.par("q").data == ["b", ""]
    assert [c.tolist() for c in result.col("x").data] == [[42], [43]]
