""" BlueSky: The open-source ATM simulator."""
# from bluesky import settings  #, stack, tools
from bluesky import settings

### Constants
SIMPLE_ECHO = 'simple_echo'
MSG_OK = 'ok.'
CMD_TCP_CONNS = 'TCP_CONNS'

### Main singleton objects in BlueSky
traf  = None
navdb = None
sim   = None
scr   = None

def init():
    # Both sim and gui need a navdatabase in all versions of BlueSky
    from bluesky.navdatabase import Navdatabase
    global navdb
    navdb = Navdatabase()

    # The remaining objects are only instantiated in the sim nodes
    if settings.is_sim:
        from bluesky.traffic import Traffic

        if settings.gui == 'pygame':
            from bluesky.ui.pygame import Screen
            from bluesky.simulation.pygame import Simulation
        else:
            from bluesky.simulation.qtgl import Simulation, ScreenIO as Screen

        from bluesky import stack
        from bluesky.tools import plugin, plotter

        # Initialize singletons
        global traf, sim, scr
        traf  = Traffic()
        sim   = Simulation()
        scr   = Screen()

        # Initialize remaining modules
        plugin.init()
        plotter.init()
        stack.init()
