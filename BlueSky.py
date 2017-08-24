#!/usr/bin/env python
""" Overall BlueSky start script """
from __future__ import print_function
from bluesky import settings

print("   *****   BlueSky Open ATM simulator *****")
print("Distributed under GNU General Public License v3")

if settings.gui == 'pygame':
    import BlueSky_pygame as bs
elif settings.gui == 'qtgl':
    import BlueSky_qtgl as bs
else:
    import sys
    print('Unknown gui type:', settings.gui)
    sys.exit(0)

# Start the main loop. When debugging in python interactive mode,
# relevant objects are available in bs namespace (e.g., bs.gui, bs.sim)
bs.main_loop()
