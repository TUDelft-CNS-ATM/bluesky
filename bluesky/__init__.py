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
mode = ''
gui = ''

# Main singleton objects in BlueSky
net = None
traf = None
navdb = None
sim = None
scr = None
server = None


def init(mode='sim', configfile=None, scenfile=None, discoverable=False,
         gui=None, detached=False, **kwargs):
    ''' Initialize bluesky modules.

        Arguments:

    '''

    # Keep track of mode and gui type.
    globals()['mode'] = mode
    globals()['gui'] = gui

    # Initialize global settings first, possibly loading a custom config file
    settings.init(configfile)

    # Initialise tools
    tools.init()

    # Load navdatabase in all versions of BlueSky
    # Only the headless server doesn't need this
    if mode == "sim" or gui is not None:
        from bluesky.navdatabase import Navdatabase
        global navdb
        navdb = Navdatabase()

    # If mode is server-gui or server-headless start the networking server
    if mode == 'server':
        global server
        from bluesky.network.server import Server
        server = Server(discoverable, configfile, scenfile)

    # The remaining objects are only instantiated in the sim nodes
    if mode == 'sim':
        from bluesky.traffic import Traffic
        from bluesky.simulation import Simulation
        if gui == 'pygame':
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
        if scenfile:
            stack.stack(f'IC {scenfile}')

    from bluesky.core import plugin
    plugin.init(mode)
    stack.init(mode)
