"""
Setup scripts for the A2-project.
"""

from setuptools import setup, find_namespace_packages

setup(
    name="tdata",
    version="0.1.0",
    description="Financial Market Data Collection and Processing",
    author="",
    author_email="",
    package_dir={"": "packages"},
    packages=find_namespace_packages(where="packages"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "pandas_ta>=0.3.14b0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0,<3.0.0",
        "beautifulsoup4>=4.12.0,<5.0.0",
        "tinkoff-investments",
        "grpcio>=1.54.0,<2.0.0",
        "protobuf>=4.23.0,<5.0.0",
    ],
    entry_points={
        "console_scripts": [
            "tdata=packages.tdata.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
setup(
    name="agent",
    version="0.1.0",
    description="RL agents for different tasks",
    author="",
    author_email="",
    package_dir={"": "packages"},
    packages=find_namespace_packages(where="packages"),
)
