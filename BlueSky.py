#!/usr/bin/env python
""" Main BlueSky start script """



import sys
from bluesky.__main__ import main


if __name__ == "__main__":
    # Run mainloop if BlueSky is called directly
    sys.exit(main())
