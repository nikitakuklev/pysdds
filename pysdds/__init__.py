__version__ = "0.1.0"
__author__ = "Nikita Kuklev"

from .readers import read


from . import _version
__version__ = _version.get_versions()['version']
