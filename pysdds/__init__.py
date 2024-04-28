__author__ = "Nikita Kuklev"

from pysdds.readers import read  # noqa: F401
from pysdds.writers import write  # noqa: F401
from pysdds.structures import SDDSFile  # noqa: F401

from . import _version

__version__ = _version.get_versions()["version"]
