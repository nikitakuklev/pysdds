__author__ = "Nikita Kuklev"

from pysdds.readers import read
from pysdds.writers import write
from pysdds.structures import SDDSFile

from . import _version
__version__ = _version.get_versions()['version']
