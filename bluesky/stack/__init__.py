''' The BlueSky command stack.

    The stack parses all text-based commands in the simulation.
'''
from bluesky.stack.stackbase import stack, forward, sender, routetosender, get_scenname, get_scendata, set_scendata
from bluesky.stack.cmdparser import command, commandgroup, append_commands, \
    remove_commands, get_commands
from bluesky.stack.argparser import refdata, ArgumentError


def init(mode='client'):
    if mode[:3] == 'sim':
        import bluesky.stack.simstack as simstack
        from bluesky.stack.importer import Importer
        simstack.init()
    else:
        import bluesky.stack.clientstack as clientstack
        clientstack.init()
