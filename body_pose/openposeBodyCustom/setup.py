from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="openpose_pytorch",  # Required
    version="0.0.0",  # Required
    packages=find_packages(where="openpose_pytorch"),  # Required
    python_requires=">=3.7, <4",
)