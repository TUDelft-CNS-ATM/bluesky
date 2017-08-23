""" BlueSky: The open-source ATM simulator."""
# from bluesky import settings  #, stack, tools
from bluesky import settings
import bluesky.traf as _traf
import bluesky.navdb as _navdb

if settings.gui == 'pygame':
    import bluesky.ui.pygame.screen as _scr
    import bluesky.sim.pygame.simulation as _sim
else:
    import bluesky.sim.qtgl.screenio as _scr
    import bluesky.sim.qtgl.simulation as _sim

### Main singleton objects in BlueSky
traf  = _traf.traffic.traf
navdb = _navdb.navdb
sim   = _sim.sim
scr   = _scr.scr

SIMPLE_ECHO = 'simple_echo'
MSG_OK = "ok."
CMD_TCP_CONNS = "TCP_CONNS"

