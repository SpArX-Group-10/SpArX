import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Sparx",
    version="1.0.0",
    author="Sparx Group",
    author_email="",
    description="Sparx Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpArX-Group-10/SpArX",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ),
)