import os
from pathlib import Path
from setuptools import find_packages, setup

install_requires = (
    Path(__file__).parent.joinpath("requirements.txt").read_text().splitlines()
)

setup(
    name="sample-reduction-with-typicality",
    version="1.0.0",
    install_requires=install_requires,
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires='>3.9.0',
)
