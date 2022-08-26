#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [Extension('cstatebased', sources=['cstatebased.cpp'])]

setup(name='cstatebased', version='1.0', include_dirs=[np.get_include(), '../../../tools/src_cpp'],
      ext_modules=ext_modules)
