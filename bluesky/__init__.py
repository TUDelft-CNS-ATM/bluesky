""" BlueSky: The open-source ATM simulator."""
# from bluesky import settings  #, stack, tools
from bluesky import settings
from bluesky.traf import Traffic
from bluesky.navdb import Navdatabase

if settings.gui == 'pygame':
    from bluesky.ui.pygame import Screen
    from bluesky.sim.pygame import Simulation
else:
    from bluesky.sim.qtgl import Simulation, ScreenIO as Screen

### Main singleton objects in BlueSky
traf  = Traffic()
navdb = Navdatabase()
sim   = Simulation()
scr   = Screen()

SIMPLE_ECHO = 'simple_echo'
MSG_OK = 'ok.'
CMD_TCP_CONNS = 'TCP_CONNS'
