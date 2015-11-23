#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

import freestream

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='freestream',
    version=freestream.__version__,
    description='Free streaming for heavy-ion collision initial conditions.',
    long_description=long_description,
    author='Jonah Bernhard',
    author_email='jonah.bernhard@gmail.com',
    url='https://github.com/Duke-QCD/freestream',
    license='MIT',
    py_modules=['freestream'],
    install_requires=['numpy>=1.8.0', 'scipy'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
