import io

import pytest
import pysdds
from pathlib import Path
import itertools
import pandas as pd
import numpy as np

cwd = Path(__file__).parent
root_sources = cwd / "files"
root_binary_rowmajor = root_sources / "sources_binary_rowmajor"
root_binary_colmajor = root_sources / "sources_binary_colmajor"
root_ascii = root_sources / "ascii"

to_str = lambda l: [str(s) for s in l]
files_sources = to_str((root_sources / "sources").glob("*"))
files_ascii = to_str((root_sources / "sources_ascii").glob("*"))
files_binary_colmajor = to_str((root_sources / "sources_binary_colmajor").glob("*"))
files_binary_rowmajor = to_str((root_sources / "sources_binary_rowmajor").glob("*"))
files_compressed = to_str((root_sources / "sources_compressed").glob("*"))
files_large = to_str((root_sources / "sources_large").glob("*"))

all_files = files_sources + files_ascii + files_binary_colmajor + files_binary_rowmajor + files_large

get_names = lambda xa: [Path(x).name for x in xa]
get_name = lambda x: Path(x).name

# Generate pairs [(file1, file1_ascii, ...), (file2, ...)] for equality testing
set_sources = set(get_names(files_sources))
set_ascii_rowmajor = set(get_names(files_ascii))
set_binary_rowmajor = set(get_names(files_binary_rowmajor))
set_binary_colmajor = set(get_names(files_binary_colmajor))

set_union = set_sources.intersection(set_ascii_rowmajor, set_binary_rowmajor, set_binary_colmajor)
sets_list = []
for f in set_sources:
    if f in set_union:
        sets_list.append(
            [str(root_sources) + f, str(root_binary_colmajor) + f, str(root_binary_rowmajor) + f, str(root_ascii) + f]
        )
    else:
        continue

# This will create AB and BA comparison to ensure things are commutative
all_tuples = [x for files in sets_list for x in itertools.product(files, repeat=2)]
all_files1 = [x[0] for x in all_tuples]
all_files2 = [x[1] for x in all_tuples]


@pytest.mark.parametrize("file_root", files_ascii)
def test_round_trip_ascii_ascii(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    # lines = buf.getvalue().decode('ascii').split('\n')
    # for l in lines:
    #     print(l)
    #     pass
    # print('--')
    sdds2 = pysdds.read(io.BufferedReader(buf))
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    buf.seek(0)
    # print(buf.getvalue().decode())
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == sdds.endianness
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_ascii(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode("ascii")
    pysdds.write(sdds, buf)
    buf.seek(0)
    #
    # print()
    # print("=====================================")
    # lines = buf.getvalue().decode('ascii').split('\n')
    # for l in lines:
    #     print(repr(l))
    #     pass
    # print("=====================================")

    # with open(file_root+'_writeback', "wb") as f:
    #     f.write(buf.getbuffer())

    sdds2 = pysdds.read(io.BufferedReader(buf))
    sdds.compare(sdds2)

    # sdds3 = pysdds.read(file_root+'_writeback')
    # sdds.compare(sdds3)
    # sdds2.compare(sdds3)
    # os.remove(os.path.abspath(file_root+'_writeback'))


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_bincol_le(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode("binary")
    sdds.set_endianness("little")
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == "little"
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_bincol_be(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode("binary")
    sdds.set_endianness("big")
    pysdds.write(sdds, buf)
    buf.seek(0)
    # lines = buf.getvalue()
    # print(lines)
    # print('--')
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == "big"
    sdds.compare(sdds2)


def test_write_from_df():
    meas_df = {
        "ControlName": ["foo-_~|aaaa", "bar\n\r\005"],
        "LowerLimit": [-2, -2],
        "UpperLimit": [-2.0, -2.0],
        "UnsignedVal": [+3, +4],
        "Description": [
            "foo-_~|aaaa",
            "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
        ],
    }
    df_meas = pd.DataFrame.from_dict(meas_df)
    df_meas["ReallyUnsigned"] = np.array([4, 4], dtype=np.uint64)
    df_meas['Horror!@#$)(*&^%#@&$#&$^@}{":?<'] = np.array([5.0, 5.1], dtype=np.float32)
    parameters = {"par1": [1], "par2": [1.0], "par3": ["foo"]}

    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode="ascii")
    assert sdds.n_pages == 1
    assert sdds.n_columns == 7
    assert sdds.n_parameters == 3
    assert sdds.columns[0].type == "string"
    assert sdds.columns[1].type == "long64"
    assert sdds.columns[2].type == "double"
    assert sdds.columns[3].type == "long64"
    assert sdds.columns[4].type == "string"
    assert sdds.columns[5].type == "ulong64"
    assert sdds.columns[6].type == "float"
    assert sdds.columns[0].data[0].dtype == object
    assert sdds.columns[1].data[0].dtype == np.int64
    assert sdds.columns[2].data[0].dtype == np.float64
    assert sdds.columns[3].data[0].dtype == np.int64
    assert sdds.columns[4].data[0].dtype == object
    assert sdds.columns[5].data[0].dtype == np.uint64
    assert sdds.columns[6].data[0].dtype == np.float32

    def run_compare(sdds):
        sdds.validate_data()
        buf = io.BytesIO()
        pysdds.write(sdds, buf)
        buf.seek(0)
        sdds2 = pysdds.read(io.BufferedReader(buf))
        sdds.compare(sdds2)
        assert np.array_equal(sdds2.columns[0].data[0], df_meas.iloc[:, 0])
        assert np.array_equal(sdds2.columns[1].data[0], df_meas.iloc[:, 1])
        assert np.array_equal(sdds2.columns[2].data[0], df_meas.iloc[:, 2])
        assert np.array_equal(sdds2.columns[3].data[0], df_meas.iloc[:, 3])
        assert np.array_equal(sdds2.columns[4].data[0], df_meas.iloc[:, 4])

    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode="ascii")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], mode="ascii")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode="binary")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], mode="binary")
    run_compare(sdds)

    sdds = pysdds.SDDSFile.from_df([df_meas], mode="binary", endianness="big")
    run_compare(sdds)
