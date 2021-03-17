''' Main stack functions. '''
import math
import os
import subprocess
import bluesky as bs
from bluesky.stack.cmdparser import Command, command, commandgroup
from bluesky.stack.basecmds import initbasecmds
from bluesky.stack import recorder
from bluesky.stack import argparser
from bluesky import settings


# Register settings defaults
settings.set_variable_defaults(start_location="EHAM", scenario_path="scenario")

# List of TMX commands not yet implemented in BlueSky
tmxlist = ("BGPASAS", "DFFLEVEL", "FFLEVEL", "FILTCONF", "FILTTRED", "FILTTAMB",
           "GRAB", "HDGREF", "MOVIE", "NAVDB", "PREDASAS", "RENAME", "RETYPE",
           "SWNLRPASAS", "TRAFRECDT", "TRAFLOGDT", "TREACT", "WINDGRID")

# Stack data
cmdstack = []  # The actual stack: Current commands to be processed

# Scenario details
scenname = ""  # Currently used scenario name (for reading)
scentime = []  # Times of the commands from the read scenario file
scencmd = []  # Commands from the scenario file
sender_rte = None  # bs net route to sender


def init(startup_scnfile):
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

    # Load initial scenario if passed
    if startup_scnfile:
        stack(f'IC {startup_scnfile}')


def reset():
    """ Reset the stack. """
    global scentime, scencmd, scenname

    scentime = []
    scencmd = []
    scenname = ""

    # Close recording file and reset scenario recording settings
    recorder.reset()
    # Reset parser reference values
    argparser.reset()


def stack(*cmdlines, cmdsender=None):
    """ Stack one or more commands separated by ";" """
    for cmdline in cmdlines:
        cmdline = cmdline.strip()
        if cmdline:
            for line in cmdline.split(";"):
                cmdstack.append((line, cmdsender))


def checkscen():
    """ Check if commands from the scenario buffer need to be stacked. """
    if scencmd:
        # Find index of first timestamp exceeding bs.sim.simt
        idx = next((i for i, t in enumerate(scentime) if t > bs.sim.simt), None)
        # Stack all commands before that time, and remove from scenario
        stack(*scencmd[:idx])
        del scencmd[:idx]
        del scentime[:idx]


def process():
    ''' Process and empty command stack. '''
    global sender_rte

    # First check for commands in scenario file
    checkscen()

    # Process stack of commands
    for line, sender_rte in cmdstack:
        success = True
        echotext = ''
        echoflags = bs.BS_OK

        # Get first argument from command line and check if it's a command
        cmd, argstring = argparser.getnextarg(line)
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
                result = cmdobj(argstring)
                if result is not None:
                    if isinstance(result, tuple) and result:
                        if len(result) > 1:
                            echotext = result[1]
                        success = result[0]
                    else:
                        # Assume result is a bool indicating the success of the function
                        success = result
                    if not success:
                        if not argstring:
                            echotext = echotext or cmdobj.brieftext()
                        else:
                            echoflags = bs.BS_FUNERR
                            echotext = f'Syntax error: {echotext or cmdobj.brieftext()}'

            except Exception as e:
                success = False
                echoflags = bs.BS_ARGERR
                header = '' if not argstring else e.args[0] if e.args else 'Argument error.'
                echotext = f'{header}\nUsage:\n{cmdobj.brieftext()}'

        # ----------------------------------------------------------------------
        # ZOOM command (or use ++++  or --  to zoom in or out)
        # ----------------------------------------------------------------------
        elif cmdu[0] in ("+", "=", "-"):
            nplus = cmdu.count("+") + cmdu.count("=")  # = equals + (same key)
            nmin = cmdu.count("-")
            bs.scr.zoom(math.sqrt(2) ** (nplus - nmin), absolute=False)
            cmdu = 'ZOOM'

        # -------------------------------------------------------------------
        # Command not found
        # -------------------------------------------------------------------
        else:
            echoflags = bs.BS_CMDERR
            if not argstring:
                echotext = f'Unknown command or aircraft: {cmd}'
            else:
                echotext = f'Unknown command: {cmd}'

        # Recording of actual validated commands
        if success:
            recorder.savecmd(cmdu, line)
        elif not sender_rte:
            echotext = f'{line}\n{echotext}'

        # Always return on command
        if echotext:
            bs.scr.echo(echotext, echoflags)

    # Clear the processed commands
    cmdstack.clear()


def sender():
    """ Return the sender of the currently executed stack command.
        If there is no sender id (e.g., when the command originates
        from a scenario file), None is returned. """
    return sender_rte[-1] if sender_rte else None


def routetosender():
    """ Return the route to the sender of the currently executed stack command.
        If there is no sender id (e.g., when the command originates
        from a scenario file), None is returned. """
    return sender_rte


def get_scenname():
    """ Return the name of the current scenario.
        This is either the name defined by the SCEN command,
        or otherwise the filename of the scenario. """
    return scenname


def get_scendata():
    """ Return the scenario data that was loaded from a scenario file. """
    return scentime, scencmd


def set_scendata(newtime, newcmd):
    """ Set the scenario data. This is used by the batch logic. """
    global scentime, scencmd
    scentime = newtime
    scencmd = newcmd


def readscn(fname):
    ''' Read a scenario file. '''
    # Split the incoming filename into a path + filename and an extension
    base, ext = os.path.splitext(fname.replace("\\", "/"))
    if not os.path.isabs(base):
        base = os.path.join(settings.scenario_path, base)
    ext = ext or ".scn"

    # The entire filename, possibly with added path and extension
    fname_full = os.path.normpath(base + ext)

    with open(fname_full, "r") as fscen:
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
    absrel = "REL"  # default relative to the time of call
    if pcall_arglst and pcall_arglst[0] in ("ABS", "REL"):
        absrel = pcall_arglst[0]
        pcall_arglst = pcall_arglst[1:]

    # If timestamps in file should be interpreted as relative we need to add
    # the current simtime to every timestamp
    t_offset = bs.sim.simt if absrel == "REL" else 0.0

    # Read the scenario file
    # readscn(fname, pcall_arglst, t_offset)
    insidx = 0
    instime = bs.sim.simt

    try:
        for (cmdtime, cmdline) in readscn(fname):

            # Time offset correction
            cmdtime += t_offset

            # Replace %0, %1 with pcall_arglst[0], pcall_arglst[1], etc.
            if pcall_arglst:
                for i, argtxt in enumerate(pcall_arglst):
                    cmdline = cmdline.replace(f"%{i}", argtxt)

            if not scentime or cmdtime >= scentime[-1]:
                scentime.append(cmdtime)
                scencmd.append(cmdline)
            else:
                if cmdtime > instime:
                    insidx, instime = next(
                        ((j, t) for j, t in enumerate(scentime) if t >= cmdtime),
                        (len(scentime), scentime[-1]),
                    )
                scentime.insert(insidx, cmdtime)
                scencmd.insert(insidx, cmdline)
                insidx += 1

        # stack any commands that are already due
        checkscen()
    except FileNotFoundError as e:
        return False, f"PCALL: File not found'{e.filename}'"


@command(aliases=('LOAD', 'OPEN'))
def ic(filename : 'string' = ''):
    ''' IC: Load a scenario file (initial condition).

        Arguments:
        - filename: The filename of the scenario to load. Call IC IC
          to load previous scenario again. '''
    global scenname

    # reset sim always
    bs.sim.reset()

    # Get the filename of new scenario
    if not filename:
        filename = bs.scr.show_file_dialog()

    # Clean up filename
    filename = filename.strip()

    # Reset sim and open new scenario file
    if filename:
        try:
            for (cmdtime, cmd) in readscn(filename):
                scentime.append(cmdtime)
                scencmd.append(cmd)
            scenname, _ = os.path.splitext(os.path.basename(filename))

            # Remember this filename in IC.scn in scenario folder
            with open(settings.scenario_path + "/" + "ic.scn", "w") as keepicfile:
                keepicfile.write(
                    "# This file is used by BlueSky to save the last used scenario file\n"
                )
                keepicfile.write(
                    "# So in the console type 'IC IC' to restart the previously used scenario file\n"
                )
                keepicfile.write("00:00:00.00>IC " + filename + "\n")

            return True, f"IC: Opened {filename}"
        except FileNotFoundError:
            return False, f"IC: File not found: {filename}"


@command(aliases=('SCEN',))
def scenario(name: 'string'):
    """ SCENARIO sets the scenario name for the current simulation.

        Arguments:
        - name: The name to give the scenario """
    global scenname
    scenname = name
    return True, "Starting scenario " + name


@command
def schedule(time: 'time', cmdline: 'string'):
    """ SCHEDULE a stack command at a specific simulation time.

        Arguments:
        - time: the time at which the command should be executed
        - cmdline: the command line to be executed """
    # Get index of first scentime greater than 'time' as insert position
    idx = next((i for i, t in enumerate(scentime) if t > time), len(scentime))
    scentime.insert(idx, time)
    scencmd.insert(idx, cmdline)
    return True


@command
def delay(time: 'time', cmdline: 'string'):
    """ DELAY a stack command until a specific simulation time.

        Arguments:
        - time: the time with which the command should be delayed
        - cmdline: the command line to be executed after the delay """
    # Get index of first scentime greater than 'time' as insert position
    time += bs.sim.simt
    idx = next((i for i, t in enumerate(scentime) if t > time), len(scentime))
    scentime.insert(idx, time)
    scencmd.insert(idx, cmdline)
    return True


@commandgroup(name='HELP', aliases=('?',))
def showhelp(cmd:'txt'='', subcmd:'txt'=''):
    """ HELP: Display general help text or help text for a specific command,
        or dump command reference in file when command is >filename.

        Arguments:
        - cmd: Command name to display help for. Call HELP >filename to generate
          a CSV file with help text for all commands.
    """
    # No command given: show all
    if not cmd:
        return True, (
            "There are different ways to get help:\n"
            " HELP PDF  gives an overview of the existing commands\n"
            " HELP cmd  gives a help line on the command (syntax)\n"
            " DOC  cmd  show documentation of a command (if available)\n"
            "And there is more info in the docs folder and the wiki on Github"
        )

    # Check if help is asked for a specific command
    cmdobj = Command.cmddict.get(cmd)
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


@showhelp.subcommand
def pdf():
    ''' Open a pdf file with BlueSky command help text. '''
    os.chdir("docs")
    pdfhelp = "BLUESKY-COMMAND-TABLE.pdf"
    if os.path.isfile(pdfhelp):
        try:
            subprocess.Popen(pdfhelp, shell=True)
        except:
            return "Opening " + pdfhelp + " failed."
    else:
        return pdfhelp + "does not exist."
    os.chdir("..")
    return "Pdf window opened"


@command
def makedoc():
    ''' MAKEDOC: Make markdown templates for all stack functions
        that don't have a doc page yet.
    '''
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    # Get unique set of commands
    cmdobjs = set(Command.cmddict.values())
    for o in cmdobjs:
        if not os.path.isfile(f"data/html/{o.name}.html"):
            with open(f"tmp/{o.name.lower()}.md", "w") as f:
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
