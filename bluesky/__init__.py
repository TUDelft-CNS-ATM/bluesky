''' BlueSky: The open-source ATM simulator.'''
from bluesky import settings
from bluesky.core import Signal
from bluesky import stack
from bluesky import tools


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
    ''' Initialize bluesky modules.

        Arguments:
        - mode: can be 'sim', 'sim-detached', 'server-gui', 'server-headless',
          or 'client'
        - pygame: indicate if BlueSky is started with BlueSky_pygame.py
        - discovery: Enable network discovery
    '''
    # Is this a server running headless?
    headless = (mode[-8:] == 'headless')

    # Keep track of the gui type.
    global gui_type
    gui_type = 'pygame' if pygame else \
               'none' if headless or mode[:3] == 'sim' else 'qtgl'

    # Initialize global settings first, possibly loading a custom config file
    settings.init(cfgfile)

    # Initialise tools
    tools.init()

    # Load navdatabase in all versions of BlueSky
    # Only the headless server doesn't need this
    if not headless:
        from bluesky.navdatabase import Navdatabase
        global navdb
        navdb = Navdatabase()

    # If mode is server-gui or server-headless start the networking server
    if mode[:6] == 'server':
        global server
        from bluesky.network.server import Server
        server = Server(discovery)

    # The remaining objects are only instantiated in the sim nodes
    if mode[:3] == 'sim':
        # Check whether simulation node should run detached
        detached = (mode[-8:] == 'detached')
        from bluesky.traffic import Traffic
        from bluesky.simulation import Simulation
        if pygame:
            from bluesky.ui.pygame import Screen
            from bluesky.network.detached import Node
        else:
            from bluesky.simulation import ScreenIO as Screen
            if detached:
                from bluesky.network.detached import Node
            else:
                from bluesky.network.node import Node

        from bluesky.core import varexplorer

        # Initialize singletons
        global traf, sim, scr, net
        traf = Traffic()
        sim = Simulation()
        scr = Screen()
        net = Node(settings.simevent_port,
                   settings.simstream_port)

        # Initialize remaining modules
        varexplorer.init()
        if scnfile:
            stack.stack(f'IC {scnfile}')

    from bluesky.core import plugin
    plugin.init(mode)
    stack.init(mode)
