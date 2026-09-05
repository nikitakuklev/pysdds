__author__ = "Nikita Kuklev"

from importlib.metadata import PackageNotFoundError, version

from pysdds.readers import read  # noqa: F401
from pysdds.structures import SDDSFile  # noqa: F401
from pysdds.writers import write  # noqa: F401

try:
    __version__ = version("pysdds")
except PackageNotFoundError:
    # package is not installed
    __version__ = "[unknown]"
