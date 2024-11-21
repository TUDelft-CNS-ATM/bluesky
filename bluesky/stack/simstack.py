''' Main simulation-side stack functions. '''
import math
from pathlib import Path
import traceback
import bluesky as bs
from bluesky.stack.stackbase import Stack, stack, checkscen, forward
from bluesky.stack.cmdparser import Command, command
from bluesky.stack.basecmds import initbasecmds
from bluesky.stack import recorder
from bluesky.stack import argparser, ArgumentError
from bluesky import settings


# Register settings defaults
settings.set_variable_defaults(start_location="EHAM", scenario_path="scenario")

# List of TMX commands not yet implemented in BlueSky
tmxlist = ("BGPASAS", "DFFLEVEL", "FFLEVEL", "FILTCONF", "FILTTRED", "FILTTAMB",
           "GRAB", "HDGREF", "MOVIE", "NAVDB", "PREDASAS", "RENAME", "RETYPE",
           "SWNLRPASAS", "TRAFRECDT", "TRAFLOGDT", "TREACT", "WINDGRID")


def init():
    """ Initialization of the default stack commands. This function is called
        at the initialization of the main simulation object."""

    # Initialise base commands
    initbasecmds()

    # Display Help text on start of program
    stack("ECHO BlueSky Console Window: Enter HELP or ? for info.\n"
          "Or select IC to Open a scenario file.")

    # Pan to initial location
    stack("PAN " + settings.start_location)
    stack("ZOOM 0.4")


def reset():
    """ Reset the stack. """

    Stack.reset()

    # Close recording file and reset scenario recording settings
    recorder.reset()
    # Reset parser reference values
    argparser.reset()


def process(from_pcall=None):
    ''' Sim-side stack processing. '''
    # First check for commands in scenario file
    if from_pcall is None:
        checkscen()

    # Process stack of commands
    for cmdline in Stack.commands(from_pcall):
        success = True
        echotext = ''
        echoflags = bs.BS_OK

        # Get first argument from command line and check if it's a command
        cmd, argstring = argparser.getnextarg(cmdline)
        cmdu = cmd.upper()
        cmdobj = Command.cmddict.get(cmdu)

        # If no function is found for 'cmd', check if cmd is actually an aircraft id
        if not cmdobj and cmdu in bs.traf.id:
            cmd, argstring = argparser.getnextarg(argstring)
            argstring = cmdu + " " + argstring
            # When no other args are parsed, command is POS
            cmdu = cmd.upper() if cmd else 'POS'
            cmdobj = Command.cmddict.get(cmdu)

        # Proceed if a command object was found
        if cmdobj:
            try:
                # Call the command, passing the argument string
                success, echotext = cmdobj(argstring)
                if not success:
                    if not argstring:
                        echotext = echotext or cmdobj.brieftext()
                    else:
                        echoflags = bs.BS_FUNERR
                        echotext = f'Syntax error: {echotext or cmdobj.brieftext()}'

            except ArgumentError as e:
                success = False
                echoflags = bs.BS_ARGERR
                header = '' if not argstring else e.args[0] if e.args else 'Argument error.'
                echotext = f'{header}\nUsage:\n{cmdobj.brieftext()}'
            except Exception as e:
                echoflags = bs.BS_FUNERR
                header = '' if not argstring else e.args[0] if e.args else 'Function error.'
                echotext = f'Error calling function implementation of {cmdu}: {header}\n' + \
                    'Traceback printed to terminal.'
                traceback.print_exc()

        # ----------------------------------------------------------------------
        # ZOOM command (or use ++++  or --  to zoom in or out)
        # ----------------------------------------------------------------------
        elif cmdu[0] in ("+", "=", "-"):
            # = equals + (same key)
            nplus = cmdu.count("+") + cmdu.count("=")
            nmin = cmdu.count("-")
            bs.scr.zoom(math.sqrt(2) ** (nplus - nmin), absolute=False)
            cmdu = 'ZOOM'

        # -------------------------------------------------------------------
        # Command not found
        # -------------------------------------------------------------------
        elif Stack.sender_rte is None:
            # Command came from scenario file: assume it's a gui/client command and send it on
            forward()
        else:
            success = False
            echoflags = bs.BS_CMDERR
            if not argstring:
                echotext = f'Unknown command or aircraft: {cmd}'
            else:
                echotext = f'Unknown command: {cmd}'

        # Recording of actual validated commands
        if success:
            recorder.savecmd(cmdu, cmdline)
        elif not Stack.sender_rte:
            echotext = f'{cmdline}\n{echotext}'

        # Always return on command
        if echotext:
            bs.scr.echo(echotext, echoflags)

    # Clear the processed commands
    if from_pcall is None:
        Stack.clear()


def readscn(fname):
    ''' Read a scenario file. '''
    if not fname:
        return
    fname = Path(fname)

    # Ensure .scn suffix and specify path if necessary
    scn_file_suffix = '.scn'
    if fname.suffix.lower() != scn_file_suffix:
        fname = fname.with_suffix(scn_file_suffix)

    if not fname.is_absolute():
        fname = bs.resource(settings.scenario_path) / fname

    with open(fname, "r") as fscen:
        prevline = ''
        for line in fscen:
            line = line.strip()
            # Skip emtpy lines and comments
            if len(line) < 12 or line[0] == "#":
                continue
            line = prevline + line

            # Check for line continuation
            if line[-1] == '\\':
                prevline = f'{line[:-1].strip()} '
                continue
            prevline = ''

            # Try reading timestamp and command
            try:
                line = line.split("#")[0].strip()

                icmdline = line.index(">")
                tstamp = line[:icmdline]
                ttxt = tstamp.strip().split(":")
                ihr = int(ttxt[0]) * 3600.0
                imin = int(ttxt[1]) * 60.0
                xsec = float(ttxt[2])
                cmdtime = ihr + imin + xsec

                yield (cmdtime, line[icmdline + 1:].strip("\n"))
            except (ValueError, IndexError):
                # nice try, we will just ignore this syntax error
                if not (len(line.strip()) > 0 and line.strip()[0] == "#"):
                    print("except this:" + line)


@command(aliases=('CALL',), brief="PCALL filename [REL/ABS/args]")
def pcall(fname, *pcall_arglst):
    """ PCALL: Import another scenario file into the current scenario.

        Arguments:
        - fname: The filename of the scenario file to import
        - REL/ABS: indicate whether timestamps in imported scenario file should
          be considered relative to current simulation time (the default),
          or absolute. (optional argument)
        - pcall_arglist: optional additional arguments to pass to the
          imported scenario. Replaces %0, %1, ... in the scenario file.
    """
    # Check for a/c id as first argument (use case: procedure files)
    # CALL KL204 myproc should have effect as if: CALL myproc KL204
    if pcall_arglst and fname in bs.traf.id:
        acid = fname
        fname = pcall_arglst[0]
        pcall_arglst = [acid] + list(pcall_arglst[1:])

    # Check for relative or absolute time
    isrelative = True  # default relative to the time of call
    if pcall_arglst and pcall_arglst[0] in ('ABS', 'REL'):
        isrelative = pcall_arglst[0] == 'ABS'
        pcall_arglst = pcall_arglst[1:]

    try:
        merge(readscn(fname), *pcall_arglst, isrelative=isrelative)

    except FileNotFoundError as e:
        return False, f"PCALL: File not found'{e.filename}'"


def merge(source, *args, isrelative=True):
    ''' Merge scenario commands from source to current scenario.'''

    # If timestamps in file should be interpreted as relative we need to add
    # the current simtime to every timestamp
    t_offset = bs.sim.simt if isrelative else 0.0

    # Read the scenario file
    insidx = 0
    instime = bs.sim.simt

    # All commands with timestamps at the current sim time or earlier should be called immediately
    callnow = []
    for (cmdtime, cmdline) in source:

        # Time offset correction
        cmdtime += t_offset

        # Replace %0, %1 with pcall_arglst[0], pcall_arglst[1], etc.
        if args:
            for i, argtxt in enumerate(args):
                cmdline = cmdline.replace(f"%{i}", argtxt)

        if cmdtime <= bs.sim.simt:
            callnow.append((cmdline, None))
        elif not Stack.scentime or cmdtime >= Stack.scentime[-1]:
            Stack.scentime.append(cmdtime)
            Stack.scencmd.append(cmdline)
        else:
            if cmdtime > instime:
                insidx, instime = next(
                    ((j, t) for j, t in enumerate(Stack.scentime) if t >= cmdtime),
                    (len(Stack.scentime), Stack.scentime[-1]),
                )
            Stack.scentime.insert(insidx, cmdtime)
            Stack.scencmd.insert(insidx, cmdline)
            insidx += 1

    # execute any commands that are already due
    if callnow:
        process(callnow)


@command(aliases=('LOAD', 'OPEN'))
def ic(filename : 'string' = ''):
    ''' IC: Load a scenario file (initial condition).

        Arguments:
        - filename: The filename of the scenario to load. Call IC IC
          to load previous scenario again. '''

    # reset sim always
    bs.sim.reset()

    # Get the filename of new scenario
    if not filename:
        filename = bs.scr.show_file_dialog()
        if not filename:
            # Only PyGame returns a filename from the dialog here
            return

    # Clean up filename
    filename = Path(filename)

    # Reset sim and open new scenario file
    try:
        for (cmdtime, cmd) in readscn(filename):
            Stack.scentime.append(cmdtime)
            Stack.scencmd.append(cmd)
        Stack.scenname = filename.stem

        # Remember this filename in IC.scn in scenario folder
        with open(bs.resource(settings.scenario_path) / "ic.scn", "w") as keepicfile:
            keepicfile.write(
                "# This file is used by BlueSky to save the last used scenario file\n"
            )
            keepicfile.write(
                "# So in the console type 'IC IC' to restart the previously used scenario file\n"
            )
            keepicfile.write(f"00:00:00.00>IC {filename}\n")

        return True, f"IC: Opened {filename}"
    except FileNotFoundError:
        return False, f"IC: File not found: {filename}"


@command(aliases=('SCEN',))
def scenario(name: 'string'):
    """ SCENARIO sets the scenario name for the current simulation.

        Arguments:
        - name: The name to give the scenario """
    Stack.scenname = name
    return True, "Starting scenario " + name


@command
def schedule(time: 'time', cmdline: 'string'):
    """ SCHEDULE a stack command at a specific simulation time.

        Arguments:
        - time: the time at which the command should be executed
        - cmdline: the command line to be executed """
    # Get index of first scentime greater than 'time' as insert position
    idx = next((i for i, t in enumerate(Stack.scentime) if t > time), len(Stack.scentime))
    Stack.scentime.insert(idx, time)
    Stack.scencmd.insert(idx, cmdline)
    return True


@command
def delay(time: 'time', cmdline: 'string'):
    """ DELAY a stack command until a specific simulation time.

        Arguments:
        - time: the time with which the command should be delayed
        - cmdline: the command line to be executed after the delay """
    # Get index of first scentime greater than 'time' as insert position
    time += bs.sim.simt
    idx = next((i for i, t in enumerate(Stack.scentime) if t > time), len(Stack.scentime))
    Stack.scentime.insert(idx, time)
    Stack.scencmd.insert(idx, cmdline)
    return True


@command(name='HELP', aliases=('?',))
def showhelp(cmd:'txt'='', subcmd:'txt'=''):
    """ HELP: Display general help text or help text for a specific command,
        or dump command reference in file when command is >filename.

        Arguments:
        - cmd: Argument can refer to:
            - Command name to display help for. 
            - Call HELP >filename to generate a CSV file with help text for all commands.
    """

    # Check if help is asked for a specific command
    cmdobj = Command.cmddict.get(cmd or 'HELP')
    if cmdobj:
        return True, cmdobj.helptext(subcmd)

    # Write command reference to tab-delimited text file
    if cmd[0] == ">":
        # Get filename
        if len(cmd) > 1:
            fname = "./docs/" + cmd[1:]
        else:
            fname = "./docs/bluesky-commands.txt"

        # Get unique set of commands
        cmdobjs = set(Command.cmddict.values())
        table = []  # for alphabetical sort use a table

        # Get info for all commands
        for obj in cmdobjs:
            fname = obj.callback.__name__.replace("<", "").replace(">", "")
            args = ','.join((str(p) for p in obj.parsers))
            syn = ','.join(obj.aliases)
            line = f'{obj.name}\t{obj.help}\t{obj.brief}\t{args}\t{fname}\t{syn}'
            table.append(line)

        # Sort & write table
        table.sort()
        with open(fname, "w") as f:
            # Header of first table
            f.write("Command\tDescription\tUsage\tArgument types\tFunction\tSynonyms\n")
            f.write('\n'.join(table))
        return True, "Writing command reference in " + fname

    return False, "HELP: Unknown command: " + cmd


@command
def makedoc():
    ''' MAKEDOC: Make markdown templates for all stack functions
        that don't have a doc page yet.
    '''
    tmp = Path('tmp')
    if not tmp.is_dir():
        tmp.mkdir()
    # Get unique set of commands
    cmdobjs = set(Command.cmddict.values())
    for o in cmdobjs:
        if not bs.resource(f"html/{o.name}.html").is_file():
            with open(tmp / f"{o.name.lower()}.md", "w") as f:
                f.write(
                    f"# {o.name}: {o.name.capitalize()}\n"
                    + o.help
                    + "\n\n"
                    + "**Usage:**\n\n"
                    + f"    {o.brief}\n\n"
                    + "**Arguments:**\n\n"
                )
                if not o.parsers:
                    f.write("This command has no arguments.\n\n")
                else:
                    f.write(
                        "|Name|Type|Optional|Description\n"
                        + "|--------|------|---|---------------------------------------------------\n"
                    )
                    for arg in o.parsers:
                        f.write(str(arg).replace(':', '|') + f" |{arg.hasdefault()}|\n")
                f.write("\n[[Back to command reference.|Command Reference]]\n")


@command(aliases=tmxlist)
def tmx(*args):
    ''' Stub function for TMX commands that aren't available yet in BlueSky. '''
    return True, 'This TMX command has not (yet) been implemented in BlueSky.'
