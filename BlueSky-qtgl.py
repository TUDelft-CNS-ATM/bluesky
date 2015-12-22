from bluesky.traf import Navdatabase
from bluesky.ui.qtgl import Gui
from bluesky.sim.qtgl import Simulation, MainLoop
from bluesky import settings
settings.gui = 'pygame'

# =============================================================================
# Create gui and simulation objects
# =============================================================================
navdb = Navdatabase('global')
gui   = Gui(navdb)
sim   = Simulation(gui,navdb)


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
