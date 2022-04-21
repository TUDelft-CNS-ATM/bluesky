''' BlueSky Stack base data and functions. '''
import bluesky as bs


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
    sender_rte = None  # bs net route to sender

    @classmethod
    def reset(cls):
        ''' Reset stack variables. '''
        cls.cmdstack = []
        cls.scenname = ""
        cls.scentime = []
        cls.scencmd = []
        cls.sender_rte = None

    @classmethod
    def commands(cls, from_pcall=None):
        ''' Generator function to iterate over stack commands. '''
        # Return commands from PCALL if passed, otherwise own command stack
        for cls.current, cls.sender_rte in from_pcall or cls.cmdstack:
            yield cls.current

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
    """ Stack one or more commands separated by ";" """
    for cmdline in cmdlines:
        cmdline = cmdline.strip()
        if cmdline:
            for line in cmdline.split(";"):
                Stack.cmdstack.append((line, sender_id))


def forward(cmd=None, *args):
    ''' Forward a stack command. 

        Sends command on to the client if this stack is running sim-side,
        and vice-versa.
    '''
    if Stack.sender_rte is None:
        # Only forward if this command originated here
        bs.net.send_event(b'STACK', f'{cmd} {",".join(args)}' if cmd else Stack.current)


def sender():
    """ Return the sender of the currently executed stack command.
        If there is no sender id (e.g., when the command originates
        from a scenario file), None is returned. """
    return Stack.sender_rte[-1] if Stack.sender_rte else None


def routetosender():
    """ Return the route to the sender of the currently executed stack command.
        If there is no sender id (e.g., when the command originates
        from a scenario file), None is returned. """
    return Stack.sender_rte


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
