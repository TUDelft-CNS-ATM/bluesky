''' BlueSky: The open-source ATM simulator.'''
from bluesky import settings
from bluesky.core import Signal
from bluesky.pathfinder import resource
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
         gui=None, detached=False, workdir=None, **kwargs):
    ''' Initialize bluesky modules.

        Arguments:
        - mode: Running mode of this bluesky process [sim/client/server]
        - configfile: Load a different configuration file [filename]
        - scenfile: Start with a running scenario [filename]
        - discoverable: Make server discoverable through UDP (only relevant 
          when this process is running a server) [True/False]
        - gui: Gui type (only when mode is client or server) [qtgl/pygame/console]
        - detached: Run with or without networking (only when mode is sim) [True/False]
        - workdir: Pass a custom working directory (instead of cwd or ~/bluesky)
    '''

    # Argument checking
    assert mode in ('sim', 'client', 'server'), f'BlueSky init: Unrecognised mode {mode}. '\
        'Possible modes are sim, client, and server.'
    assert gui in (None, 'qtgl', 'pygame', 'console'), f'BlueSky init: Unrecognised gui type {gui}. '\
        'Possible types are qtgl, pygame, and console.'
    if discoverable:
        assert mode == 'server', 'BlueSky init: Discoverable can only be set in server mode.'
    if scenfile:
        assert mode != 'client', 'BlueSky init: Scenario file cannot be passed to a client.'
    if gui:
        assert mode != 'sim' or gui == 'pygame', 'BlueSky init: Gui type shouldn\'t be specified in sim mode.'
    if detached:
        assert mode == 'sim', 'BlueSky init: Detached operation is only available in sim mode.'

    # Keep track of mode and gui type.
    globals()['mode'] = mode
    globals()['gui'] = gui

    # Initialise resource localisation, and set custom working directory if present
    from bluesky import pathfinder
    pathfinder.init(workdir)

    # Initialize global settings, possibly loading a custom config file
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
