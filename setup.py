import os
from pathlib import Path
from setuptools import find_packages, setup

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
package_path = Path(__file__).parent
data_files = [
    str(version_path.relative_to(package_path)),
]
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
    data_files=data_files,
    python_requires='>3.9.0',
)
