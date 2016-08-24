# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:47:44 2016

@author: metz_is
"""

import pstats
p=pstats.Stats('wasbeer.txt')
p.strip_dirs().sort_stats(-1).print_stats()

