""" BlueSky: The open-source ATM simulator."""
from bluesky import settings


# Constants
BS_OK = 0
BS_ARGERR = 1
BS_FUNERR = 2
BS_CMDERR = 4

# simulation states
INIT, HOLD, OP, END = list(range(4))

# Startup flags
gui_type = ''
startup_scnfile = ''

# Main singleton objects in BlueSky
net = None
traf = None
navdb = None
sim = None
scr = None
server = None


def init(mode='sim', pygame=False, discovery=False, cfgfile='', scnfile=''):
    """ Initialize bluesky modules.

        Arguments:
        - mode: can be 'sim', 'sim-detached', 'server-gui', 'server-headless',
          or 'client'
        - pygame: indicate if BlueSky is started with BlueSky_pygame.py
        - discovery: Enable network discovery
    """
    # Initialize global settings first, possibly loading a custom config file
    settings.init(cfgfile)

    # Is this a server running headless?
    headless = (mode[-8:] == 'headless')

    # Keep track of the gui type.
    global gui_type
    gui_type = 'pygame' if pygame else \
               'none' if headless or mode[:3] == 'sim' else 'qtgl'

    # Load navdatabase in all versions of BlueSky
    # Only the headless server doesn't need this
    if not headless:
        from bluesky.navdatabase import Navdatabase
        global navdb
        navdb = Navdatabase()

    # If mode is server-gui or server-headless start the networking server
    if mode[:6] == 'server':
        global server
        from bluesky.network import Server
        server = Server(headless)

    # The remaining objects are only instantiated in the sim nodes
    if mode[:3] == 'sim':
        # Check whether simulation node should run detached
        detached = (mode[-8:] == 'detached')
        from bluesky.traffic import Traffic

        if pygame:
            from bluesky.ui.pygame import Screen
            from bluesky.simulation.pygame import Simulation
        else:
            from bluesky.simulation.qtgl import Simulation, ScreenIO as Screen

        from bluesky import stack
        from bluesky.tools import plugin, varexplorer

        # Initialize singletons
        global traf, sim, scr
        traf = Traffic()
        sim = Simulation(detached)
        scr = Screen()

        # Initialize remaining modules
        plugin.init(mode)
        varexplorer.init()
        stack.init(scnfile)
