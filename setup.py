#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:51:40 2023

max scharf Sa 14. Jan 15>52>16 CET 2023 maximilian.scharf_at_uol.de
Pip install the package in editable state with pip install -e <myproject_folder> to keep on working on this package


see https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/
"""

from setuptools import setup, find_packages

setup(name='consitencyMeasure',
      version='1.0.0', # see: https://peps.python.org/pep-0440/
      packages=find_packages(),
      description="A measure of consitency for psychometric data",
      long_description="This package quantifies how likely psychometric measurements are caused by a single psychometric function",
      long_description_content_type="text/plain",
      author="M.K. Scharf",
      author_email="maximilian.scharf@uni-oldenburg.de",
      license="GPLv3",
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
    
        # Indicate who your project is intended for
        'Intended Audience :: Researcher',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
    
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',],
      install_requires=["matplotlib",
      			"numpy",
                        "pathlib",
                        "scipy",
                        ],
      python_requires='>=3')
