from setuptools import setup
try:
    import versioneer
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import versioneer

setup(version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass())
