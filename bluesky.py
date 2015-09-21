import sys

from bluesky.ui import Gui
from bluesky.sim import Simulation, MainLoop

# =============================================================================
# Create gui and simulation objects
# =============================================================================
gui = Gui(sys.argv)
sim = Simulation()


# =============================================================================
# Start the mainloop (and possible other threads)
# =============================================================================
MainLoop(gui, sim)

# =============================================================================
# Clean up before exit. Comment this out when debugging for checking variables
# in the shell.
# =============================================================================
del gui
#-debug del sim

print 'BlueSky normal end.'
