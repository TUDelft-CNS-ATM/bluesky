from bluesky import settings
settings.gui = 'qtgl'

from bluesky.traf import Navdatabase
from bluesky.ui.qtgl import Gui
from bluesky.sim.qtgl import Simulation, Thread


# =============================================================================
# Start the mainloop (and possible other threads)
# =============================================================================
def MainLoop():
    # =============================================================================
    # Create gui and simulation objects
    # =============================================================================
    navdb = Navdatabase('global') # Read database from specified folder
    gui   = Gui(navdb)
    sim   = Simulation(gui,navdb)

    # Create a simulation thread, and start it at the highest priority
    simthread = Thread(sim)
    simthread.start(Thread.HighestPriority)

    # Set sim.screenio as an event target for the gui, 
    # so that the gui can send events to the sim object
    gui.setSimEventTarget(sim.screenio)

    # Start the gui
    gui.start()

    # Stopping simulation thread
    print 'Stopping Threads'
    sim.stop()
    simthread.quit()
    simthread.wait()

    return gui


if __name__ == "__main__":
    # Run mainloop if BlueSky-qtgl is called directly
    gui = MainLoop()

    # =============================================================================
    # Clean up before exit. Comment this out when debugging for checking variables
    # in the shell.
    # =============================================================================
    del gui
    #-debug del sim
    print 'BlueSky normal end.'
