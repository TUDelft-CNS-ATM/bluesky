#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [Extension('casas', sources=['casas.cpp'])]

setup(name='casas', version='1.0', include_dirs=[np.get_include(), '../../../tools/src_cpp'],
      ext_modules=ext_modules)
