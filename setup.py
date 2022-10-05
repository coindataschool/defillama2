#!/usr/bin/env python3

from setuptools import setup

VERSION = '1.0'

packages = ['defillama2']
requires = ['requests>=2.28.1', 'pandas>=1.4.4', 'numpy>=1.22.4', 'dateutil>=2.8.2']

setup(
    name="defillama2", 
    version=VERSION,
    description='Python client for DefiLlama APIs',
    long_description='Download and clean data from DefiLlama.com.',
    url='https://github.com/coindataschool/defillama',
    author="Coin Data School",
    author_email="<coindataschool@gmail.com>",
    packages=packages,
    install_requires=requires, # dependencies    
    keywords=['python 3', 'defillama', 'api'],
    classifiers= [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)