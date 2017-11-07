#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as f:
    long_description = f.read()

def version():
    with open('freestream.py', 'r') as f:
        for l in f:
            if l.startswith('__version__'):
                return l.split('=')[1].strip(" '\n")

setup(
    name='freestream',
    version=version(),
    description='Free streaming for heavy-ion collision initial conditions.',
    long_description=long_description,
    author='Jonah Bernhard',
    author_email='jonah.bernhard@gmail.com',
    url='https://github.com/Duke-QCD/freestream',
    license='MIT',
    py_modules=['freestream'],
    install_requires=['numpy>=1.8.0', 'scipy>=0.14.0'],
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
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
