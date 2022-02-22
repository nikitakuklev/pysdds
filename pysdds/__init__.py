__author__ = "Nikita Kuklev"

from .readers import read
from .writers import write

from . import _version
__version__ = _version.get_versions()['version']
