from pathlib import Path
import subprocess

import bluesky as bs
from bluesky.stack.stackbase import Stack, forward, stack
from bluesky.stack.cmdparser import Command, command, commandgroup
from bluesky.stack import argparser

def init():
    ''' client-side stack initialisation. '''
    pass


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

        elif Stack.sender_rte is None:
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
            bs.scr.echo(echotext, echoflags, Stack.sender_rte)

    # Clear the processed commands
    Stack.cmdstack.clear()


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
    bs.net.send_event(b'STACK', f'HELP {cmd} {subcmd}')


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
