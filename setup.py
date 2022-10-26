#!/usr/bin/env python3

from setuptools import setup

VERSION = '0.6'

packages = ['defillama2']
requires = ['requests>=2.28.1', 'pandas>=1.4.4', 'numpy>=1.22.4', 
    'python-dateutil>=2.8.2']

with open('README.md', mode='r') as f:
    readme = f.read()

setup(
    name="defillama2", 
    version=VERSION,
    description='Python client for DefiLlama API',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/coindataschool/defillama2',
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