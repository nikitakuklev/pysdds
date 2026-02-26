__author__ = "Nikita Kuklev"

from pysdds.readers import read  # noqa: F401
from pysdds.writers import write  # noqa: F401
from pysdds.structures import SDDSFile  # noqa: F401

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pysdds")
except PackageNotFoundError:
    # package is not installed
    __version__ = "[unknown]"
