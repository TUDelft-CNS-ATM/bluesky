""" BlueSky: The open-source ATM simulator."""
# from bluesky import settings  #, stack, tools
from bluesky import settings

### Constants
BS_OK = 0
BS_ARGERR = 1
BS_FUNERR = 2
BS_CMDERR = 4

# simulation states
INIT, HOLD, OP, END = list(range(4))

### Main singleton objects in BlueSky
net = None
traf = None
navdb = None
sim = None
scr = None
server = None

def init():
    # Both sim and gui need a navdatabase in all versions of BlueSky
    if settings.is_sim or settings.is_gui:
        from bluesky.navdatabase import Navdatabase
        global navdb
        navdb = Navdatabase()

    if settings.start_server:
        global server
        from bluesky.network import Server
        server = Server()
        server.start()

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
