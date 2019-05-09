#!/usr/bin/env python
""" Main BlueSky start script """
from __future__ import print_function
import sys
import traceback
import bluesky as bs

# Create custom system-wide exception handler. For now it replicates python's
# default traceback message. This was added to counter a new PyQt5.5 feature
# where unhandled exceptions would result in a qFatal with a very uninformative
# message.
def exception_handler(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit()


sys.excepthook = exception_handler


def main():
    """
        Start BlueSky: This is the main entrypoint for BlueSky.
        Depending on settings and arguments passed it can start in different
        modes. The central part of BlueSky consists of a server managing all
        simulations, normally together with a gui. The different modes for this
        are:
        - server-gui: Start gui and simulation server
        - server-headless: start server without gui
        - client: start gui only, which can connect to an already running server

        A BlueSky server can start one or more simulation processes, which run
        the actual simulations. These simulations can also be started completely
        separate from all other BlueSky functionality, in the detached mode.
        This is useful when calling bluesky from within another python
        script/program. The corresponding modes are:
        - sim: The normal simulation process started by a BlueSky server
        - sim-detached: An isolated simulation node, without networking
    """
    # When importerror gives different name than (pip) install needs,
    # also advise latest version
    missingmodules = {"OpenGL": "pyopengl and pyopengl-accelerate"}

    ### Parse command-line arguments ###
    # BlueSky.py modes:
    # server-gui: Start gui and simulation server
    # client: start gui only, which can connect to an already running server
    # server-headless: start server only
    # detached: start only one simulation node, without networking
    #   ==> useful for calling bluesky from within another python script/program
    if '--detached' in sys.argv:
        mode = 'sim-detached'
    elif '--sim' in sys.argv:
        mode = 'sim'
    elif '--client' in sys.argv:
        mode = 'client'
    elif '--headless' in sys.argv:
        mode = 'server-headless'
    else:
        mode = 'server-gui'

    discovery = ('--discoverable' in sys.argv or mode[-8:] == 'headless')

    # Check if alternate config file is passed or a default scenfile
    cfgfile = ''
    scnfile = ''
    for i in range(len(sys.argv)):
        if len(sys.argv) > i + 1:
            if sys.argv[i] == '--config-file':
                cfgfile = sys.argv[i + 1]
            elif sys.argv[i] == '--scenfile':
                scnfile = sys.argv[i + 1]

    # Catch import errors
    try:
        # Initialize bluesky modules
        bs.init(mode, discovery=discovery, cfgfile=cfgfile, scnfile=scnfile)

        # Only start a simulation node if called with --sim or --detached
        if mode[:3] == 'sim':
            if mode[-8:] != 'detached':
                bs.sim.connect()
            bs.sim.run()
        else:
            # Only print start message in the non-sim cases to avoid printing
            # this for every started node
            print("   *****   BlueSky Open ATM simulator *****")
            print("Distributed under GNU General Public License v3")

        # Start server if server/gui or server-headless is started here
        if mode[:6] == 'server':
            if mode[-8:] == 'headless':
                bs.server.run()
            else:
                bs.server.start()

        # Start gui if client or main server/gui combination is started here
        if mode in ('client', 'server-gui'):
            from bluesky.ui import qtgl
            qtgl.start(mode)

    # Give info on missing module
    except ImportError as error:
        modulename = missingmodules.get(error.name) or error.name
        if modulename is None:
            raise error
        print("Bluesky needs", modulename)
        print("Install using e.g. pip install", modulename)

    print('BlueSky normal end.')


if __name__ == "__main__":
    # Run mainloop if BlueSky is called directly
    main()
