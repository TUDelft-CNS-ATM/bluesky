''' The BlueSky command stack.

    The stack parses all text-based commands in the simulation.
'''
from bluesky.stack.stackbase import stack, forward, sender, get_scenname, get_scendata, set_scendata
from bluesky.stack.cmdparser import command, commandgroup, append_commands, \
    remove_commands, get_commands
from bluesky.stack.argparser import ArgumentError


def process(ext_cmds):
    ''' Dummy process that will get replaced by sim or client implementation during init. '''
    pass


def init(mode='client'):
    if mode[:3] == 'sim':
        import bluesky.stack.simstack as simstack
        from bluesky.stack.importer import Importer
        simstack.init()
        globals()['process'] = simstack.process
    else:
        import bluesky.stack.clientstack as clientstack
        clientstack.init()
        globals()['process'] = clientstack.process
