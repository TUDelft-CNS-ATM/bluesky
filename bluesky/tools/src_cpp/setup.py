#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [Extension('cgeo', sources=['cgeo.cpp'])]

setup(name='cgeo', version='1.0', include_dirs=[np.get_include()],
      ext_modules=ext_modules)
