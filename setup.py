import os
import setuptools

current_dir = os.path.dirname(os.path.abspath(__file__))

readme_path = os.path.join(current_dir, "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements_path = os.path.join(current_dir, "requirements.txt")
with open(requirements_path, "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="sparx-lib",
    version="1.0.1",
    author="Sparx Group",
    author_email="",
    description="Sparx Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpArX-Group-10/SpArX",
    packages=["sparx"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
)
