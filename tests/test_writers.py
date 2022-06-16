import io

import pytest
import pysdds
import glob
from pathlib import Path
import itertools
import pandas as pd
import numpy as np

root_sources = 'files/sources/'
root_binary_rowmajor = 'files/binary_rowmajor/'
root_binary_colmajor = 'files/binary_colmajor/'
root_ascii = 'files/ascii/'

files_sources = glob.glob(root_sources + '*')
files_ascii = glob.glob(root_ascii + '*')
files_binary_colmajor = glob.glob(root_binary_colmajor + '*')
files_binary_rowmajor = glob.glob(root_binary_rowmajor + '*')
files_compressed = glob.glob('files/sources_compressed/*')
files_large = glob.glob('files/sources_large/*')

all_files = files_sources + files_ascii + files_binary_colmajor + files_binary_rowmajor + files_large

get_names = lambda xa: [Path(x).name for x in xa]
get_name = lambda x: Path(x).name

# Generate pairs [(file1, file1_ascii, ...), (file2, ...)] for equality testing
set_sources = set(get_names(files_sources))
set_ascii_rowmajor = set(get_names(files_ascii))
set_binary_rowmajor = set(get_names(files_binary_rowmajor))
set_binary_colmajor = set(get_names(files_binary_colmajor))

set_union = set_sources.intersection(set_ascii_rowmajor).intersection(set_binary_rowmajor).intersection(
    set_binary_colmajor)
sets_list = []
for f in set_sources:
    if f in set_union:
        sets_list.append([root_sources + f, root_binary_colmajor + f, root_binary_rowmajor + f, root_ascii + f])
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
    pysdds.write(sdds, buf, endianness=sdds.endianness)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == sdds.endianness
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_ascii(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode('ascii')
    pysdds.write(sdds, buf)
    buf.seek(0)
    #
    print()
    print("=====================================")
    lines = buf.getvalue().decode('ascii').split('\n')
    for l in lines:
        print(repr(l))
        pass
    print("=====================================")
    #
    # with open(file_root+'_writeback', "wb") as f:
    #     f.write(buf.getbuffer())
    #
    sdds2 = pysdds.read(io.BufferedReader(buf))
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_bincol_le(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode('binary')
    sdds.set_endianness('little')
    pysdds.write(sdds, buf)
    buf.seek(0)
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == 'little'
    sdds.compare(sdds2)


@pytest.mark.parametrize("file_root", files_sources)
def test_round_trip_sources_bincol_be(file_root):
    sdds = pysdds.read(file_root)
    buf = io.BytesIO()
    sdds.set_mode('binary')
    sdds.set_endianness('big')
    pysdds.write(sdds, buf)
    buf.seek(0)
    # lines = buf.getvalue()
    # print(lines)
    # print('--')
    sdds2 = pysdds.read(io.BufferedReader(buf))
    assert sdds2.endianness == 'big'
    sdds.compare(sdds2)


def test_write_from_df():
    meas_df = {'ControlName': ['foo', 'bar'], 'LowerLimit': [-2, -2], 'UpperLimit': [-2.0, -2.0]}
    df_meas = pd.DataFrame.from_dict(meas_df)
    parameters = {'par1': [1], 'par2': [1.0], 'par3': ['foo']}
    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode='ascii')
    sdds.validate_data()
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
    sdds = pysdds.SDDSFile.from_df([df_meas], parameter_dict=parameters, mode='binary')
    sdds.validate_data()
    buf = io.BytesIO()
    pysdds.write(sdds, buf)
