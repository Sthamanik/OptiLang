from setuptools import setup, find_packages

"""
This file exists for the backward compatibility
All configuration is in pyproject.toml
"""

setup(packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]))
