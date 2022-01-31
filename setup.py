import setuptools
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

INSTALL_REQUIRES = [
    'numpy>=1.13.3',
    'pandas>=1.0'
]
EXTRAS_REQUIRES = {
    "develop": [
        "pytest>=6.0",
    ]
}
LICENSE = 'MIT'
DESCRIPTION = 'Package to read and write SDDS files'
CLASSIFIERS = [
    "Programming Language :: Python :: 3.8",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
]

setuptools.setup(
    name="pysdds",
    author="Nikita Kuklev",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=setuptools.find_packages(where="pysdds"),
    package_dir={"": "pysdds"},
    description=DESCRIPTION,
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    platforms="any",
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.8",
    extras_require=EXTRAS_REQUIRES
)
