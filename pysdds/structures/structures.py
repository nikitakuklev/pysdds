__all__ = ['Description', 'Parameter', 'Array', 'Column', 'Data', 'SDDSFile']

import logging
import math
import sys
from typing import List, Optional, Literal, Dict

import numpy as np
import pandas as pd

from ..util import constants

logger = logging.getLogger(__name__)


def _compare_arrays(one, two, eps=None) -> bool:
    if isinstance(one, np.ndarray) and isinstance(two, np.ndarray):
        if one.dtype != two.dtype:
            raise Exception(f'Dtypes dont match??? {repr(one.dtype)} {repr(two.dtype)}')
        assert len(one) == len(two)
        if np.issubdtype(one.dtype, np.number) and eps is not None:
            # Numeric
            assert np.issubdtype(two.dtype, np.number)
            return np.allclose(one, two, atol=eps, rtol=0.0, equal_nan=True)
        else:
            # Not numeric
            # for i in range(len(one)):
            #     if one[i] != two[i]:
            #         return False
            # return True
            # return np.array_equal(one, two)
            if np.issubdtype(one.dtype, np.dtype('str_')):
                for i in range(len(one)):
                    if one[i] != two[i]:
                        return False
                return True
            else:
                return np.all(np.equal(one, two))
    elif isinstance(one, list) and isinstance(two, list):
        raise Exception
        for i in range(len(one)):
            if not math.isclose(one[i], two[i], abs_tol=eps, rel_tol=0.0):
                return False
        return True
    else:
        raise Exception(f'Comparison of two different types - {type(one)} vs {type(two)}')


def _find_different_indices(one, two):
    idxs = []
    for i,(x,y) in enumerate(zip(one, two)):
        if x != y:
            idxs.append(i)
    return idxs


def _namelist_to_str(nm_dict):
    kv_strings = []
    for k, v in nm_dict.items():
        if isinstance(v, str) and (' ' in v or '"' in v or "," in v or "$" in v):
            kv_strings.append(f'{k}="{v}"')
        else:
            kv_strings.append(f'{k}={v}')
    return ", ".join(kv_strings)


class Description:
    """
    Data Set Description

    &description
        STRING text = NULL
        STRING contents = NULL
    &end

    This optional command describes the data set in terms of two strings. The first, text, is an informal description
    that is intended principly for human consumption. The second, contents, is intended to formally specify the type
    of data stored in a data set. Most frequently, the contents field is used to record the name of the program that
    created or most recently modified the file.
    """

    def __init__(self, namelist):
        self.nm = namelist

    def __eq__(self, other):
        return self.compare(other)

    def __str__(self):
        return f'&description {_namelist_to_str(self.nm)}, &end'

    def to_sdds(self):
        return f'&description {_namelist_to_str(self.nm)}, &end'

    def compare(self, other: 'Description', raise_error: bool = False) -> bool:
        """ Compare to another object """
        fail_str = f'SDDSFile {self} mismatch: '

        def err(stage, *args):
            if raise_error:
                raise Exception(fail_str + stage + ' ' + '|'.join([str(a) for a in args]))

        if other is None:
            err('None comparison')
            return False
        if type(self) != type(other):
            err('type')
            return False
        if self.nm != other.nm:
            err('namelist')
            return False
        return True

    @property
    def text(self):
        return self.nm.get('text', None)

    @property
    def contents(self):
        return self.nm.get('contents', None)


class Parameter:
    """
    Parameter Definition

    &parameter
        STRING name = NULL
        STRING symbol = NULL
        STRING units = NULL
        STRING description = NULL
        STRING format_string = NULL
        STRING type = NULL
        STRING fixed_value = NULL
    &end

    This optional command defines a parameter that will appear along with the tabular data section of each data page.
    The name field must be supplied, as must the type field. The type must be one of short, long, float, double,
    character, or string, indicating the corresponding C data types. The string type refers to a NULL-terminated
    character string.
    """

    __slots__ = ('__dict__', 'nm', 'sdds')

    def __init__(self, namelist, sdds: "SDDSFile" = None):
        self.nm = namelist
        # Data list is a list of values, 1 per page
        # It will generated dynamically for fixed value parameters
        # if self.fixed_value is None:
        #    self.data = []
        self._data = []
        self.__cached_data = None
        self.__cached_page_count = None
        self.sdds = sdds

    def __str__(self):
        if self.data:
            return f'Parameter "{self.name}" ({len(self.data)} pages) at <{hex(id(self))}>: {self.nm}'
        else:
            return f'Parameter "{self.name}" (empty) at <{hex(id(self))}>: {self.nm}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: "Parameter"):
        return self.compare(other)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise ValueError('Only integer indexing is allowed for parameters')
        if key >= len(self._data) or key < 0:
            raise KeyError(f'Index {key} invalid for data length {len(self._data)}')
        return self._data[key]

    def to_sdds(self):
        return f'&parameter {_namelist_to_str(self.nm)},  &end'

    def compare(self, other, eps: Optional[float] = None, raise_error: bool = False,
                fixed_equivalent: bool = True) -> bool:
        """ Compare to another object based solely on data and not layout """
        fail_str = f'Parameter {self.name} mismatch: '

        def err(stage, *args):
            if raise_error:
                raise Exception(fail_str + stage + ' ' + '|'.join([repr(a) for a in args]))

        if type(self) != type(other):
            err('type', type(self), type(other))
            return False
        if self.nm != other.nm:
            if fixed_equivalent:
                keys = set(self.nm.keys()).union(set(other.nm.keys())) - {'fixed_value'}
                for k in keys:
                    if k not in self.nm or k not in other.nm or self.nm[k] != other.nm[k]:
                        err('nm', self.nm, other.nm)
                        return False
            else:
                err('nm', self.nm, other.nm)
                return False
        if self.fixed_value is not None and other.fixed_value is not None:
            # Only 1 value
            if self.type in ['string', 'char'] or eps is None:
                if self.fixed_value != other.fixed_value:
                    err('strict fixed_value')
                    return False
            else:
                if not math.isclose(self.fixed_value, other.fixed_value, abs_tol=eps, rel_tol=0.0):
                    err(f'eps fixed_value')
                    return False
        # elif ((self.fixed_value is not None and other.fixed_value is None) or
        #       (self.fixed_value is None and other.fixed_value is not None)) and fixed_equivalent:
        #     # self is fixed, other is not
        #     if self.fixed_value is not None:
        #         fixed = self.fixed_value
        #         variable = other.data
        #     else:
        #         fixed = other.fixed_value
        #         variable = self.data
        #
        #     if self.type in ['string', 'char'] or eps is None:
        #         for i in range(len(variable)):
        #             if variable[i] != fixed:
        #                 err('strict', variable, fixed)
        #                 return False
        #     else:
        #         for i in range(len(variable)):
        #             if not math.isclose(variable[i], fixed, abs_tol=eps, rel_tol=0.0):
        #                 # if not _compare_arrays(self.data, other.data, eps):
        #                 err('eps', variable, fixed)
        #                 return False
        else:
            # Full parameter
            if len(self.data) != len(other.data):
                return False
            if self.type in ['string', 'character'] or eps is None:
                for i in range(len(self.data)):
                    if self.data[i] != other.data[i]:
                        err(f'strict', self.data[i], other.data[i])
                        return False
            else:
                for i in range(len(self.data)):
                    if not math.isclose(self.data[i], other.data[i], abs_tol=eps, rel_tol=0.0):
                        # if not _compare_arrays(self.data, other.data, eps):
                        err(f'eps', self.data, other.data)
                        return False
        return True

    @property
    def name(self):
        return self.nm.get('name', None)

    @property
    def type(self):
        return self.nm.get('type', None)

    @property
    def fixed_value(self):
        return self.nm.get('fixed_value', None)

    @property
    def data(self):
        """
        Data property requires additional logic to return lists for fixed value parameters
        """
        if self.fixed_value is not None:
            if self.__cached_page_count != self.sdds.n_pages:
                self.__cached_data = [self.fixed_value for _ in range(self.sdds.n_pages)]
                self.__cached_page_count = self.sdds.n_pages
            return self.__cached_data
        else:
            return self._data

    @data.setter
    def data(self, value):
        if self.fixed_value is not None:
            raise ValueError(
                f'Attempted to set data for parameter ({self.name}), but it already has fixed value ({self.fixed_value})')
        else:
            assert isinstance(value, list)
            self._data = value


class Array:
    """
    Array Definition

    &array
        STRING name = NULL
        STRING symbol = NULL
        STRING units = NULL
        STRING description = NULL
        STRING format_string = NULL
        STRING type = NULL
        STRING group_name = NULL
        long field_length = 0
        long dimensions = 1
    &end

    This optional command defines an array that will appear along with the tabular data section of each data page. The
    name field must be supplied, as must the type field. The type must be one of short, ushort, long, ulong, float,
    double, longdouble, character, or string, indicating the corresponding C data types. The string type refers to a
    NULL-terminated character string.

    The optional symbol field allows specification of a symbol to represent the array;
    it may contain escape sequences, for example, to produce Greek or mathematical characters. The optional units field
    allows specification of the units of the array. The optional description field provides for an informal description
    of the array. The optional format_string field allows specification of the printf format string to be used to print
    the data (e.g., for ASCII in SDDS or other formats). The optional group_name field allows specification of a string
    giving the name of the array group to which the array belongs; such strings may be defined by the user to indicate
    that different arrays are related (e.g., have the same dimensions, or parallel elements). The optional dimensions
    field gives the number of dimensions in the array.

    The order in which successive array commands appear is the order
    in which the arrays are assumed to come in the data. For ASCII data, each array will occupy at least one line in the
    input file ahead of the tabular data; data for different arrays may not occupy portions of the same line.
    """

    __slots__ = ('__dict__', 'nm', 'data', 'sdds')

    def __init__(self, namelist, sdds: "SDDSFile" = None):
        self.nm = namelist
        self.data = []
        self._page_numbers = None
        self._enabled = True
        self.sdds = sdds

    def __eq__(self, other: "Array"):
        return self.compare(other, eps=None)

    def to_sdds(self):
        return f'&array {_namelist_to_str(self.nm)},  &end'

    def compare(self, other: "Array", eps: Optional[float] = None, raise_error: bool = False) -> bool:
        fail_str = f'Array {self.name} mismatch: '

        def err(stage, *args):
            if raise_error:
                raise Exception(fail_str + stage + ' ' + '|'.join([str(a) for a in args]))

        """ Compare to another Array """
        if type(self) != type(other):
            err('type')
            return False
        if self.nm != other.nm:
            err('nm')
            return False
        # if self._page_numbers != other._page_numbers:
        #     err('page_numbers')
        #     return False
        if len(self.data) != len(other.data):
            err('array length')
            return False
        if self._enabled != other._enabled:
            err('enabled')
            return False
        # Check lengths as first, cheap step
        # for i in range(len(self._page_numbers)):
        #     if len(self.data[i]) != len(other.data[i]):
        #         err('data length', len(self.data[i]), len(other.data[i]))
        #         return False
        for i in range(len(self.data)):
            if len(self.data[i]) != len(other.data[i]):
                err('data length', len(self.data[i]), len(other.data[i]))
                return False
        # Walk the data
        for i in range(len(self.data)):
            if eps is not None:
                if not _compare_arrays(self.data[i], other.data[i], eps):
                    err('values eps', self.data[i], other.data[i])
                    return False
            else:
                if not _compare_arrays(self.data[i], other.data[i]):
                    err('values strict', self.data[i], other.data[i])
                    return False
        return True

    @property
    def name(self):
        return self.nm.get('name', None)

    @property
    def type(self):
        return self.nm.get('type', None)

    @property
    def dimensions(self):
        return self.nm.get('dimensions', 1)

    def get(self, page: int):
        return self.data[page]


class Column:
    """
    Array Data Definition

    &array
        STRING name = NULL
        STRING symbol = NULL
        STRING units = NULL
        STRING description = NULL
        STRING format_string = NULL
        STRING type = NULL
        STRING group_name = NULL
        long field_length = 0
        long dimensions = 1
    &end

    This optional command defines an array that will appear along with the tabular data section of each data page.
    The name field must be supplied, as must the type field. The type must be one of short, long, float, double,
    character, or string, indicating the corresponding C data types. The string type refers to a NULL-terminated
    character string.
    """

    __slots__ = ('__dict__', 'nm', 'data', 'sdds')

    def __init__(self, namelist, sdds: "SDDSFile" = None):
        self.nm: dict = namelist
        self.data: List[np.ndarray] = []
        # Internal object - stores original page indices with possible discontinuities
        self._page_numbers: List[int] = []
        # Stores if column is enabled (i.e. was parsed)
        self._enabled: bool = True
        # Parent SDDSFile
        self.sdds: SDDSFile = sdds

    def __eq__(self, other: "Column"):
        return self.compare(other, eps=None)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= len(self.data) or key < 0:
                raise KeyError(f'Index {key} invalid for data length {len(self.data)}')
            return self.data[key]
        elif isinstance(key, tuple):
            assert len(key) == 2, f'Tuple {key} index length != 2'
            assert all(isinstance(k, int) for k in key), f'Not all indices in tuple {key} are integers'
            page_data = self.data[key[0]]
            if key[1] >= len(page_data) or key[1] < 0:
                raise KeyError(f'Index {key} invalid - page {key[0]} length is {len(page_data)}')
            return page_data[key[1]]
        else:
            raise ValueError(f'Only integer or integer tuple indexing is allowed for columns')

    def to_sdds(self):
        return f'&column {_namelist_to_str(self.nm)},  &end'

    def compare(self, other: "Column", eps: Optional[float] = None, raise_error: bool = False) -> bool:
        """ Compare to another Column, optionally with tolerance and other options """
        fail_str = f'Column {self.name} mismatch: '

        def err(stage, *args):
            if raise_error:
                raise Exception(fail_str + stage + ' ' + '|'.join([str(a) for a in args]))

        if type(self) != type(other):
            err('type')
            return False
        if self.nm != other.nm:
            err('nm')
            return False
        if self._page_numbers != other._page_numbers:
            err('page_numbers', self._page_numbers, other._page_numbers)
            return False
        if self._enabled != other._enabled:
            err('enabled')
            return False
        # Check lengths as first, cheap step
        for i in range(len(self._page_numbers)):
            if len(self.data[i]) != len(other.data[i]):
                err('data length', i, len(self.data[i]), len(other.data[i]))
                return False
        # Walk the data
        for i in range(len(self._page_numbers)):
            if eps is not None:
                if not _compare_arrays(self.data[i], other.data[i], eps):
                    different_idxs = _find_different_indices(self.data[i], other.data[i])
                    logger.error(f'Comparison at page {i} had diffences at indices {different_idxs}')
                    logger.error(f'Value tuples: {[(self.data[i][j], other.data[i][j]) for j in different_idxs]}')
                    err('values eps', i, self.data[i].dtype, other.data[i].dtype, self.data[i], other.data[i])
                    return False
            else:
                if not _compare_arrays(self.data[i], other.data[i]):
                    err(f'{self.name} values strict', i, self.data[i], other.data[i])
                    return False
        return True

    @property
    def name(self):
        return self.nm.get('name', None)

    @property
    def type(self):
        return self.nm.get('type', None)

    def get(self, page: int):
        return self.data[page]

    def __str__(self):
        if self.data:
            return f'Column "{self.name}" ({len(self.data)} pages) at <{hex(id(self))}>: {self.nm}'
        else:
            return f'Column "{self.name}" (empty) at <{hex(id(self))}>: {self.nm}'

    def __repr__(self):
        return self.__str__()


class Data:
    """
    Data Mode and Arrangement Definition

    &data
        STRING mode = "binary"
        long lines_per_row = 1
        long no_row_counts = 0
        long column_major_order = 0
        long additional_header_lines = 0
        [UNDOCUMENTED] STRING endian = 'little'
    &end

    This command is optional unless parameter commands without fixed_value fields, array commands, or column commands
    have been given.
    """

    __slots__ = ('__dict__', 'nm', 'data', 'sdds')

    def __init__(self, namelist=None):
        self.nm = namelist or {}

    def __eq__(self, other):
        return self.compare(other)

    def to_sdds(self):
        return f'&data {_namelist_to_str(self.nm)}, &end'

    def compare(self, other, raise_error: bool = False) -> bool:
        fail_str = f'Data mismatch: '

        def err(stage, *args):
            if raise_error:
                raise Exception(fail_str + stage + ' ' + '|'.join([str(a) for a in args]))

        if type(self) != type(other):
            err('type')
            return False
        if self.nm != other.nm:
            err('nm')
            return False
        return True

    @property
    def mode(self):
        """ The mode field is required, and may have one of the values “ascii” or “binary”. If binary mode
    is specified, the other entries of the command are irrelevant and are ignored. In ASCII mode, these entries are
    optional."""
        return self.nm.get('mode', 'binary')

    @property
    def additional_header_lines(self):
        """ If additional_header_lines is set to a non-zero value, it gives the number of non-SDDS data lines that
        follow the data command. Such lines are treated as comments. """
        return self.nm.get('additional_header_lines', 0)

    @property
    def lines_per_row(self):
        """ In ASCII mode, each row of the tabular data occupies lines_per_row rows in the file. If lines_per_row is
        zero, however, the data is assumed to be in “stream” format, which means that line breaks are irrelevant.
        Each line is processed until it is consumed, at which point the next line is read and processed. """
        return self.nm.get('lines_per_row', 1)

    @property
    def column_major_order(self):
        """ If column_major_order is set to a non-zero value and mode is set to “binary”, it will store the column
        data in column major order instead of the default row major order which normally results in faster reading
        and writing of the data file. """
        return self.nm.get('column_major_order', 0)

    @property
    def no_row_counts(self):
        """ Normally, each data page includes an integer specifying the number of rows in the tabular data section.
        This allows for preallocation of arrays for data storage, and obviates the need for an end-of-page indicator.
        However, if no_row_counts is set to a non-zero value, the number of rows will be determined by looking for
        the occurence of an empty line. A comment line does not qualify as an empty line in this sense. """
        return self.nm.get('no_row_counts', 0)


class SDDSFile:
    """
    A Python storage class representing a self-describing data set (SDDS) file

    Holds all objects in sorted lists and provides convenience access and setting methods
    """

    __slots__ = ('__dict__', 'description', 'parameters', 'arrays', 'columns', 'data', 'mode',
                 'endianness', 'n_pages')

    def __init__(self):
        self.description: Optional[Description] = None
        self.parameters: List[Parameter] = []
        self.arrays: List[Array] = []
        self.columns: List[Column] = []
        self.data: Optional[Data] = None

        self.mode: Literal["binary", "ascii"] = 'binary'
        self.endianness: Literal["big", "little"] = 'little'

        self._source_file: Optional[str] = None
        self._source_file_size: Optional[int] = None
        # self.columns_dict = None

        self.n_pages: int = 0
        self.n_parameters: int = 0
        self.n_arrays: int = 0
        self.n_columns: int = 0

        # This indicates parsing mode with more rows specified than actually exist, used in loggers
        self._meta_fixed_rowcount = False

    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            # Assume both column and page are specified
            page = keys[0]
            column = keys[1]
            if not isinstance(page, int):
                raise ValueError(f'First index is expected to be an integer corresponding to page number')

            if not 0 <= page <= self.n_pages - 1:
                raise KeyError(f'Page {page} is not within acceptable bounds (0 - {self.n_pages})')

            if not isinstance(column, str):
                raise ValueError(f'Second index is expected to be a string denoting the column')

            if column not in self.column_names:
                raise KeyError(f'Column {column} is not found (have {self.column_names})')

            return self.col(column).data[page]
        elif isinstance(keys, str):
            if keys in self.column_names:
                return self.columns_dict[keys]
            elif keys in self.array_dict:
                return self.array_dict[keys]
            elif keys in self.parameter_dict:
                return self.parameter_dict[keys]
            else:
                raise KeyError(f'Key {keys} is not found, have {self.column_names}|'
                               f'{self.array_names}|{self.parameter_names}')
        else:
            raise Exception

    def __eq__(self, other):
        """
        Two files are considered same if all data matches exactly
        See compare() for comparison with tolerance
        """
        self.compare(other, eps=None)

    def describe(self):
        line = ''
        line += f'SDDSFile: {self.n_pages} pages, {len(self.parameters)} parameters, {len(self.arrays)} arrays,' \
                f' {len(self.columns)} columns, mode {self.mode}, endianness {self.endianness}\n'
        columns = [c for c in self.columns if c._enabled]
        if len(columns) > 0:
            data = columns[0].data
            line += f' Page sizes: {[len(v) for v in data]}\n'
            bytes_used = 0
            for c in columns:
                for v in c.data:
                    bytes_used += v.nbytes
            line += f' Column mem usage: {bytes_used / 1024 / 1024:.4f} MB'
        return line

    def compare(self, other,
                eps: Optional[float] = None,
                ignore_data_mode: bool = True,
                fixed_value_equivalent: bool = False,
                raise_error: bool = False) -> bool:
        """
        Compares this SDDS file with another, allowing small discrepancies.

        Parameters
        ----------

        other : object
            The object to compare to
        eps : float
            Tolerance of numerical values
        ignore_data_mode : bool
            Treat binary and ascii data objects as same
        raise_error : bool
            If True, exception is raised when first mismatch is found

        Returns
        -------
        is_match: bool
            Whether two data structures are equivalent
        """
        fail_str = f'SDDSFile {self} mismatch: '

        def err(stage, *args):
            if raise_error:
                raise Exception(fail_str + stage + ' ' + '|'.join([str(a) for a in args]))

        if not isinstance(other, SDDSFile):
            err('type', type(self), type(other))
            return False
        if self.n_pages != other.n_pages:
            err('n_pages', self.n_pages, other.n_pages)
            return False
        if self.n_parameters != other.n_parameters:
            err('n_parameters', self.n_parameters, other.n_parameters)
            return False
        if self.n_arrays != other.n_arrays != len(self.arrays) != len(other.arrays):
            err('n_arrays', self.n_arrays, other.n_arrays)
            return False
        if self.n_columns != other.n_columns != len(self.columns) != len(other.columns):
            err('n_columns', self.n_columns, other.n_columns)
            return False

        if self.description is None:
            if other.description is not None:
                err('description None', self.description, other.description)
                return False
        else:
            if not self.description.compare(other.description, raise_error):
                return False

        for i in range(len(self.parameters)):
            # if (('fixed_value' in self.parameters[i].nm and 'fixed_value' not in other.parameters[i].nm) or
            #         ('fixed_value' in other.parameters[i].nm and 'fixed_value' not in self.parameters[i].nm) and
            #         fixed_value_equivalent):
            #     if not self.parameters[i].compare(other.parameters[i], eps, raise_error, fixed_equivalent=True):
            #         return False
            # else:
            if not self.parameters[i].compare(other.parameters[i], eps, raise_error,
                                              fixed_equivalent=fixed_value_equivalent):
                return False

        for i in range(len(self.arrays)):
            if not self.arrays[i].compare(other.arrays[i], eps, raise_error):
                return False

        for i in range(len(self.columns)):
            if not self.columns[i].compare(other.columns[i], eps, raise_error):
                return False

        if self.data is None:
            if other.data is not None:
                err('data None', self.data, other.data)
                return False
        else:
            if ignore_data_mode:
                if other.data is None:
                    err('other None', self.data, other.data)
                    return False
                else:
                    pass
            else:
                if not self.data.compare(other.data):
                    return False
        return True

    @property
    def parameter_dict(self):
        return {el.nm['name']: el for el in self.parameters}

    @property
    def parameter_names(self):
        return [el.nm['name'] for el in self.parameters]

    @property
    def array_dict(self):
        return {el.nm['name']: el for el in self.arrays}

    @property
    def array_names(self):
        return [el.nm['name'] for el in self.arrays]

    def array(self, array_name):
        contents = {el.name: el for el in self.arrays}
        return contents[array_name]

    @property
    def columns_dict(self):
        # Deprecated
        return {c.nm['name']: c for c in self.columns}

    @property
    def column_dict(self):
        return self.columns_dict

    @property
    def column_names(self):
        return [c.nm['name'] for c in self.columns]

    # @columns.setter
    # def columns(self, value):
    #     self._columns = value
    #     #self._columns_dict = {c.name: c for c in value} if value is not None else None

    def col(self, column):
        cd = {c.name: c for c in self.columns}
        return cd[column]

    def par(self, parameter):
        cd = {el.name: el for el in self.parameters}
        return cd[parameter]

    def set_mode(self, mode: Literal["binary", "ascii"], **kwargs):
        """ Set SDDS file mode, affecting how file will be written. Data namelist will be modified if present. """
        if mode not in ['ascii', 'binary']:
            raise Exception
        self.mode = mode
        if self.data is not None:
            if mode == 'binary':
                self.data.nm.update({'mode': mode, 'endian': self.endianness})
            else:
                self.data.nm.update({'mode': mode, **kwargs})

    def set_endianness(self, endianness: Literal["big", "little"]):
        if endianness == self.endianness:
            if self.mode == 'binary' and self.data is not None and 'endian' in self.data.nm:
                assert self.data.nm['endian'] == endianness
        else:
            self.endianness = endianness
            if self.mode == 'binary' and self.data is not None:
                self.data.nm['endian'] = endianness

    @staticmethod
    def from_df(df_list: List[pd.DataFrame],
                parameter_dict: Optional[Dict[str, list]] = None,
                mode: Literal["binary", "ascii"] = 'binary',
                endianness: Literal["big", "little"] = None) -> "SDDSFile":

        """
        Create SDDS object from lists of dataframes

        Parameters
        ----------
        df_list : List of dataframes
            List of dataframes to use for data, with same columns in each. Not copy-safe.
        parameter_dict : dict
            Dictionary with keys representing parameter names and values containing arrays of data, 1 per page.
            Length of each array must match that of df_list.
        mode : str
            SDDS mode
        endianness : str
            SDDS endianness

        Returns
        -------
        sdds : SDDSFile
            New SDDS object
        """

        if endianness is None or endianness == 'auto':
            endianness = sys.byteorder
        if endianness not in ['auto', 'big', 'little']:
            raise ValueError(f'SDDS binary endianness ({endianness}) is not recognized')

        page_idx = 0
        n_pages = len(df_list)
        df = df_list[page_idx]
        sdds = SDDSFile()
        columns = df.columns

        for i, c in enumerate(columns):
            if df.dtypes[i] == np.dtype(np.int64):
                val = df.iloc[:, i].values.astype(np.int32)
            else:
                val = df.iloc[:, i].values
            sdds_type = constants._NUMPY_DTYPES_INV[df.dtypes[i]]
            #print(df.dtypes[i], sdds_type)
            namelist = {'name': c, 'type': sdds_type}
            col = Column(namelist, sdds)
            sdds.columns.append(col)
            col.data.append(val)

        if parameter_dict is not None:
            for i, (k, v) in enumerate(parameter_dict.items()):
                assert len(v) == n_pages, f'Length {len(v)} of parameter {k} not equal to df list length {n_pages}'
                namelist = {'name': k, 'type': constants._PYTHON_TYPE_INV[type(v[0])]}
                par = Parameter(namelist, sdds)
                sdds.parameters.append(par)
                if type(v[0]) == str:
                    par.data = list(np.array(v, dtype=object))
                else:
                    arr = np.array(v)
                    if arr.dtype == np.dtype(np.int64):
                        arr = arr.astype(np.int32)
                    par.data = list(arr)

        for page_idx in range(1, len(df_list)):
            for i, c in enumerate(columns):
                if df.dtypes[i] == np.dtype(np.int64):
                    val = df.iloc[:, i].to_numpy(np.int32)
                else:
                    val = df.iloc[:, i].to_numpy()
                assert sdds.columns[i].data[0].dtype == val.dtype
                sdds.columns[i].data.append(val)

        sdds.data = Data({'mode': mode})
        sdds.n_columns = len(sdds.columns)
        sdds.n_parameters = len(sdds.parameters)
        sdds.set_mode(mode)
        sdds.n_pages = n_pages
        return sdds

    def columns_to_df(self, page: int = 0) -> pd.DataFrame:
        """
        Retrieve a copy of column data in specific page as a pandas dataframe

        Parameters
        ----------
        page: int
            Page number to retrieve, by default the first available page.

        Returns
        -------
        df: DataFrame
            Pandas dataframe
        """
        assert isinstance(page, int)
        if not 0 <= page < self.n_pages:
            raise ValueError(f'Page ({page}) is not valid - have ({self.n_pages})')
        columns = [c for c in self.columns if c._enabled]
        column_names = [c.name for c in columns]
        # index = np.arange(1, self.n_pages+1)
        data = {c.name: c.data[page] for c in columns}
        df = pd.DataFrame(data=data, columns=column_names)
        return df

    def parameters_to_df(self) -> pd.DataFrame:
        """
        Retrieve parameters as a pandas dataframe. Indices correspond to pages, and column labels to parameter names.

        Returns
        -------
        df : DataFrame
        """
        columns = [p.name for p in self.parameters]
        # index = np.arange(1, self.n_pages + 1)
        data = {p.name: p.data for p in self.parameters}
        df = pd.DataFrame(data=data, columns=columns)
        df.index.name = 'Page'
        return df

    def page_to_df(self, page=0) -> pd.DataFrame:
        """
        Transforms parameters and columns from single page into a single dataframe, expanding parameter values to
        columns

        Parameters
        ----------
        page : int
            Page number

        Returns
        -------
            df : DataFrame
        """
        assert isinstance(page, int)
        assert page <= self.n_pages
        columns = [c for c in self.columns if c._enabled]
        column_names = [c.name for c in columns]
        parameters = [p for p in self.parameters]
        parameter_names = [p.name for p in parameters]
        page_size = len(columns[0].data[page])
        # index = np.arange(1, self.n_pages+1)
        data = {c.name: c.data[page] for c in columns}
        data.update({p.name: np.full(page_size, p.data[page]) for p in parameters})
        df = pd.DataFrame(data=data, columns=column_names + parameter_names)
        return df

    def validate_data(self):
        """ Validate current data for self-consistency """
        n_pages = self.n_pages
        from ..util.constants import _NUMPY_DTYPE_FINAL, _PYTHON_TYPE_FINAL

        assert self.n_parameters == len(self.parameters)
        assert self.n_arrays == len(self.arrays)
        assert self.n_columns == len(self.columns)

        for el in self.parameters:
            data = el.data
            assert len(data) == n_pages
            assert isinstance(data, list)
            for v in data:
                if type(v) != _PYTHON_TYPE_FINAL[el.type]:
                    raise Exception(f'Parameter type {type(v)} ({v}) does not match {_PYTHON_TYPE_FINAL[el.type]}')

        for el in self.arrays:
            data = el.data
            assert len(data) == n_pages
            assert all(isinstance(v, np.ndarray) for v in data)
            assert all(v.dtype == _NUMPY_DTYPE_FINAL[el.type] for v in data)

        for el in self.columns:
            data = el.data
            assert len(data) == n_pages
            assert all(isinstance(v, np.ndarray) for v in data)
            for v in data:
                expected_dtype = _NUMPY_DTYPE_FINAL[el.type]
                if not v.dtype == _NUMPY_DTYPE_FINAL[el.type]:
                    if (np.issubdtype(v.dtype, np.integer) and
                        np.issubdtype(_NUMPY_DTYPE_FINAL[el.type], np.integer)):
                        # If both integer-like, probably ok since converts up to max values
                        pass
                    else:
                        raise Exception(f'dtype {v.dtype} does not match expected {expected_dtype}')
