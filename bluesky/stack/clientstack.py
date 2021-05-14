import bluesky as bs
from bluesky.stack.stackbase import Stack, stack
from bluesky.stack.cmdparser import Command, command, commandgroup
from bluesky.stack import argparser


def init():
    ''' client-side stack initialisation. '''
    pass


def process():
    ''' Client-side stack processing. '''
    # Process stack of commands
    for line, sender_rte in Stack.cmdstack:
        Stack.sender_rte = sender_rte
        success = True
        echotext = ''
        echoflags = bs.BS_OK

        # Get first argument from command line and check if it's a command
        cmd, argstring = argparser.getnextarg(line)
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

        elif sender_rte is None:
            # If sender_id is None, this stack command originated from the gui. Send it on to the sim
            bs.net.send_event(b'STACKCMD', line)
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
            bs.scr.echo(echotext, echoflags, sender_rte)

    # Clear the processed commands
    Stack.cmdstack.clear()
