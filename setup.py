"""
Setup scripts for the A2-project.
"""

from setuptools import setup, find_namespace_packages

setup(
    name="tdata-agent",
    version="0.1.0",
    description="Financial Market Data Collection, Processing, and RL Agents",
    author="",
    author_email="",
    package_dir={"": "packages"},
    packages=find_namespace_packages(where="packages"),
)
