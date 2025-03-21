"""Setup file for the causal inference project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="exoplanet-causal-inference",
    version="0.1.0",
    author="",
    author_email="",
    description="Causal inference analysis of stellar age effects on exoplanet sizes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "exoplanet-causal=src.main:main",
            "exoplanet-dash=src.dashboard_dash.app:main",
        ],
    },
) 