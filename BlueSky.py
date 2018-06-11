#!/usr/bin/env python
""" Overall BlueSky start script """
from __future__ import print_function
from bluesky import settings
import bluesky as bs

print("   *****   BlueSky Open ATM simulator *****")
print("Distributed under GNU General Public License v3")

if settings.gui == 'pygame':
    from BlueSky_pygame import main
elif settings.gui == 'qtgl':
    from BlueSky_qtgl import main
else:
    import sys
    print('Unknown gui type:', settings.gui)
    sys.exit(0)

if __name__ == '__main__':
    # Start the main loop. When debugging in python interactive mode,
    # relevant objects are available in bs namespace (e.g., bs.scr, bs.sim)
    main()
