from bluesky import settings
import sys


if settings.gui is 'qtgl':
    from bluesky.ui.qtgl import Gui
    from bluesky.sim.mt import Simulation, MainLoop
elif settings.gui is 'pygame':
    from bluesky.ui.pygame import Gui
    from bluesky.sim.st import Simulation, MainLoop


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
