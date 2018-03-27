#!/usr/bin/env python
# -*- coding: utf8 -*-
""" BlueSky start script for packaged installation. """
from __future__ import print_function
from bluesky import settings

print("   *****   BlueSky Open ATM simulator *****")
print("Distributed under GNU General Public License v3")

if settings.gui == 'pygame':
    from BlueSky_pygame import start, cleanup
elif settings.gui == 'qtgl':
    from BlueSky_qtgl import start, cleanup
else:
    import sys
    print('Unknown gui type:', settings.gui)
    sys.exit(0)

if __name__ == '__main__':
    # Start the main loop. When debugging in python interactive mode,
    # relevant objects are available in bs namespace (e.g., bs.scr, bs.sim)
    start()

    cleanup()
