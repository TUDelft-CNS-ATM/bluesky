from pathlib import Path
import subprocess
import traceback
import math

import bluesky as bs
from bluesky.core.signal import Signal
from bluesky.stack.stackbase import Stack, forward, stack
from bluesky.stack.cmdparser import Command, command, commandgroup
from bluesky.stack import argparser


# Globals
_sig_echo = Signal('echo')


def init():
    ''' client-side stack initialisation. '''
    # Display Help text on start of program
    stack("ECHO BlueSky Console Window: Enter HELP or ? for info.\n"
          "Or select IC to Open a scenario file.")


def process():
    ''' Client-side stack processing. '''
    # Process stack of commands
    for cmdline in Stack.commands():
        success = True
        echotext = ''
        echoflags = bs.BS_OK

        # Get first argument from command line and check if it's a command
        cmd, argstring = argparser.getnextarg(cmdline)
        cmdu = cmd.upper()
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

            except Exception as e:
                success = False
                echoflags = bs.BS_ARGERR
                header = '' if not argstring else e.args[0] if e.args else 'Argument error.'
                echotext = f'{header}\nUsage:\n{cmdobj.brieftext()}'
                traceback.print_exc()

        # ----------------------------------------------------------------------
        # ZOOM command (or use ++++  or --  to zoom in or out)
        # ----------------------------------------------------------------------
        elif cmdu[0] in ("+", "=", "-"):
            # = equals + (same key)
            nplus = cmdu.count("+") + cmdu.count("=")
            nmin = cmdu.count("-")
            fac = math.sqrt(2) ** (nplus - nmin)
            cmdu = 'ZOOM'
            cmdobj = Command.cmddict.get(cmdu)
            if cmdobj:
                cmdobj(f'IN {fac}')

        elif Stack.sender_id is None:
            # If sender_id is None, this stack command originated from the gui. Send it on to the sim
            forward()
        # -------------------------------------------------------------------
        # Command not found
        # -------------------------------------------------------------------
        else:
            success = False
            echoflags = bs.BS_CMDERR
            if not argstring:
                echotext = f'Unknown command or aircraft: {cmd}'
            else:
                echotext = f'Unknown command: {cmd}'

        # Always return on command
        if echotext:
            echo(echotext, echoflags, Stack.sender_id)

    # Clear the processed commands
    Stack.cmdstack.clear()


@command(annotations='string')
def echo(text='', flags=0, sender_id=None):
        ''' Echo

            Clien-side implementation of ECHO emits the same
            signal as the one triggered by incoming echo
            messages.    
        '''
        _sig_echo.emit(text, flags, sender_id)


@commandgroup(name='HELP', aliases=('?',))
def showhelp(cmd: 'txt' = '', subcmd: 'txt' = ''):
    """ HELP: Display general help text or help text for a specific command,
        or dump command reference in file when command is >filename.

        Arguments:
        - cmd: Argument can refer to:
            - Command name to display help for. 
            - Call HELP >filename to generate a CSV file with help text for all commands.
            - Call HELP PDF to view a pdf containing information on all stack commands.

        To get more detailed information on a command type DOC cmd.
    """

    # Check if help is asked for a specific command
    cmdobj = Command.cmddict.get(cmd or 'HELP')
    if cmdobj:
        return True, cmdobj.helptext(subcmd)

    # If command is not a known Client command pass the help request on to the sim
    forward(target_id=bs.net.act_id)


@showhelp.subcommand
def pdf():
    ''' Open a pdf file with BlueSky command help text. '''
    pdfhelp = Path("BLUESKY-COMMAND-TABLE.pdf")
    if (Path('docs') / pdfhelp).is_file():
        try:
            subprocess.Popen(pdfhelp, shell=True, cwd='docs')
        except:
            return "Opening " + pdfhelp.as_posix() + " failed."
    else:
        return pdfhelp.as_posix() + "does not exist."

    return "Pdf window opened"


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
                if not o.params:
                    f.write("This command has no arguments.\n\n")
                else:
                    f.write(
                        "|Name|Type|Optional|Description\n"
                        + "|--------|------|---|---------------------------------------------------\n"
                    )
                    for arg in o.params:
                        f.write(str(arg).replace(':', '|') + f" |{arg.hasdefault()}|\n")
                f.write("\n[[Back to command reference.|Command Reference]]\n")
    # To also get all of the sim stack commands, forward to sim
    forward()