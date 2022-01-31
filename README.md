<br />
<div align="center">
  <h3 align="center">pySDDS</h3>
  <p align="center">
    Pure Python SDDS reader and writer
  </p>
  ![GitHub](https://img.shields.io/github/license/nikitakuklev/pysdds)
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#advanced-considerations">Advanced considerations</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#version-history">Version history</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About
Self Describing Data Set (SDDS) [file format](https://ops.aps.anl.gov/manuals/sdds/SDDS.html) is a common format used in accelerator physics, notably by `elegant` simulation code and other [tools](https://www.aps.anl.gov/Accelerator-Operations-Physics/Software) developed at Argonne National Laboratory. _pysdds_ is a pure-Python SDDS parser and writer with several nifty features that help with integration into standard Python ML and data analysis workflows.

## Getting started
### Prerequisites
* Python >= 3.8
* numpy >= 1.19
* pandas >=1.0

### Installation
Clone this repository:
```bash
git clone https://github.com/nkuklev/pysdds
```
and add to path during runtime:
```python
sys.path.append(<location where repo was cloned>/pysdds)
```
If you have access to ANL APS network, you can install package in development mode, and it will be updated automatically with new versions:
```bash
pip install -e /home/oxygen/NKUKLEV/software/pysdds
```
Public PyPI/conda packages will be added once the codebase matures

## Usage
Reading a file:
```python
import pysdds

# All parameters are automatically inferred
sdds = pysdds.read('file.sdds')

# Compressed files will be decompressed on the fly
sdds = pysdds.read('file.sdds.xz')

# All reader options can be explicitly provided
sdds = pysdds.read('file.sdds.xz', mode='binary', compression='xz', endianness='little')

# Select only the necessary arrays, columns, and pages - other data is discarded without memory allocation
sdds = pysdds.read('file.sdds', pages=[0,2], columns=['x','y'], arrays=['matrix1'])
```

Working with data:
```python
sdds = pysdds.read('file.sdds')

# Get column 'x' and parameter 'betax'
column = sdds.col('x')
parameter = sdds.par('betax')

# NumPy array of values for column 'x' on page 0
column.data[0]

# Single value for parameter 'betax' on page 3
parameter.data[3]

# Convert page to a dataframe with appropriate column labels
sdds.columns_to_df(page=1)

# Convert all parameters to a dataframe, indexed by page number
sdds.parameters_to_df()
```

See `demos` directory for more examples.

## Advanced considerations
_pysdds_ is designed for maximum performance (insofar as possible in Python) without depending on any compiled code beyond standard library and `numpy`/`pandas`). It is an independent implementation based on SDDS specification, not associated with the official [SDDSPython3](https://www.aps.anl.gov/Accelerator-Operations-Physics/Software) package.

Major differences are:
- Pure Python, and platform agnostic (official package is a thin wrapper for core C library)
- Comparable read performance for numerical data, but **slower** for text data. You should do your own benchmarks if performance is critical.
- Intelligent file buffering - helps with network file systems
- Large file streaming without full memory pre-allocation
- Serializable data structures (for example allowing data passing during batch computations through Dask)
- Easier debugging and partial parsing of corrupted/abnormal data

Package architecture is straightforward - file header is first parsed to determine overall structure. Then, dispatch method picks one of multiple reader and writer implementations depending on file contents and compatibility constraints. Custom implementations can be provided as well (might perform better for your particular use case).

## Roadmap
- [ ] ASCII readers
  - [x] Line by line 
  - [ ] Streaming
  - [x] Text block
- [x] Binary readers
  - [x] Streaming
  - [x] Binary block via numpy
  - [x] Binary block via struct
- [ ] Writers
  - [ ] ASCII
  - [ ] Binary
- [x] Tests
  - [x] Read tests
  - [x] Compressed file tests
  - [x] Automated file gen for various format permutations

## Version history

* 0.11
  * Implemented fast zero-copy path for numeric-only data
* 0.1
  * Initial release, read support for most ASCII/binary cases

## Contributing

Pull requests are welcome, but current codebase is expected to evolve quickly until `1.0` release. It is probably more useful to just submit a feature request.

Should you decide to contribute, please ensure that all tests pass and that your code conforms to [Google Python styleguide](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings) with exception of comments, which should follow NumPy convention.

## License

This project has been developed as personal helper tool for various accelerator physics research, and is released under a permissive MIT license. Notably however, I am not responsible if this breaks your shiny particle accelerator toys.

[license-url]: https://github.com/nkuklev/pysdds/blob/master/LICENSE
