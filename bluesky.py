import sys

from bluesky.ui import Gui
from bluesky.sim import Simulation, MainLoop

# =============================================================================
# Create gui and simulation objects
# =============================================================================
MainLoop.gui = Gui(sys.argv)
MainLoop.sim = Simulation()

# =============================================================================
# Start the mainloop (and possible other threads)
# =============================================================================
MainLoop.start()

print 'BlueSky normal end.'
