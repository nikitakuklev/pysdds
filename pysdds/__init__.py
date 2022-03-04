__author__ = "Nikita Kuklev"

from .readers import read
from .writers import write
from .structures import SDDSFile

from . import _version
__version__ = _version.get_versions()['version']
