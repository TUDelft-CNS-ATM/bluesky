''' BlueSky Stack base data and functions. '''
from itertools import repeat

import bluesky as bs
from bluesky.network import subscriber, context as ctx
from bluesky.stack.cmdparser import command


class Stack:
    ''' Stack static-only namespace. '''

    # Stack data
    current = ''
    cmdstack = []  # The actual stack: Current commands to be processed

    # Scenario details
    scenname = ""  # Currently used scenario name (for reading)
    scentime = []  # Times of the commands from the read scenario file
    scencmd = []  # Commands from the scenario file

    # Current command details
    sender_id = None  # bs net route to sender

    @classmethod
    def reset(cls):
        ''' Reset stack variables. '''
        cls.cmdstack = []
        cls.scenname = ""
        cls.scentime = []
        cls.scencmd = []
        cls.current = ''
        cls.sender_id = None

    @classmethod
    def commands(cls, ext_cmds=None):
        ''' Generator function to iterate over stack commands. '''
        # Return explicitly passed commands if given, otherwise own command stack
        if isinstance(ext_cmds, str):
            # If argument is a single string convert it to an appropriate list
            ext_cmds = zip(ext_cmds.split(';'), repeat(cls.sender_id))

        for cls.current, cls.sender_id in ext_cmds or cls.cmdstack:
            yield cls.current
        # After processing commands, current command and sender id should be reset
        cls.current = ''
        cls.sender_id = None

    @classmethod
    def clear(cls):
        cls.cmdstack.clear()


def checkscen():
    """ Check if commands from the scenario buffer need to be stacked. """
    if Stack.scencmd:
        # Find index of first timestamp exceeding bs.sim.simt
        idx = next((i for i, t in enumerate(
            Stack.scentime) if t > bs.sim.simt), None)
        # Stack all commands before that time, and remove from scenario
        stack(*Stack.scencmd[:idx])
        del Stack.scencmd[:idx]
        del Stack.scentime[:idx]


def stack(*cmdlines, sender_id=None):
    """ Stack one or more commands.
        Stacking multiple commands can be done as multiple comma-separated
        arguments, and/or semicolon-separated within a single string. """
    for cmdline in cmdlines:
        cmdline = cmdline.strip()
        if cmdline:
            for line in cmdline.split(";"):
                Stack.cmdstack.append((line, sender_id))


def forward(*cmdlines, target_id=None):
    ''' Forward one or more stack commands. 

        By default, forward() sends command on to the client if this stack is
        running sim-side, and vice-versa. Instead, a target id can be specified
        to specifically send the stack command(s) to this node.
    
        Multiple commands can be specified as multiple comma-separated
        arguments, and/or semicolon-separated within a single string.
    '''
    if target_id is not None or Stack.sender_id is None:
        # Only forward if this command originated here
        bs.net.send(b'STACK', ';'.join(cmdlines) if cmdlines else Stack.current, target_id)


def sender():
    """ Return the sender of the currently executed stack command.
        If there is no sender id (e.g., when the command originates
        from a scenario file), None is returned. """
    return Stack.sender_id


def get_scenname():
    """ Return the name of the current scenario.
        This is either the name defined by the SCEN command,
        or otherwise the filename of the scenario. """
    return Stack.scenname


def get_scendata():
    """ Return the scenario data that was loaded from a scenario file. """
    return Stack.scentime, Stack.scencmd


def set_scendata(newtime, newcmd):
    """ Set the scenario data. This is used by the batch logic. """
    Stack.scentime = newtime
    Stack.scencmd = newcmd


# Register subscriber for stack commands coming from the network
@subscriber(topic='STACK')
def on_stack_received(data):
    """ Add stack commands coming from the network to the stack. """
    stack(data, sender_id=ctx.sender_id)


@command
def stackat(targetid, cmdline: 'string'):
    """ Send a stack command to a specific node (client or simulation) or group. """
    forward(cmdline, target_id=targetid)