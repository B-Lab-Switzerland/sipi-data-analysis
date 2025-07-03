from setuptools import setup, find_packages

setup(
    name='sipi_da_utils',
    version='0.1.0',
    description='Common utility functions for multiple dataset analyses in the context of SIPI',
    author='Mischa Knabenhans',
    author_email='mischa@blab-switzerland.ch',
    packages=find_packages(include=['sipi_da_utils', 'sipi_da_utils.*'])
)