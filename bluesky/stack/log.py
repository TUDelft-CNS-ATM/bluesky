# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:55:16 2014

@author: Jerom Maas
"""
import sys
sys.path.append('bluesky/tools/')
import random
import numpy as np
from aero import ft, nm, eas2tas

def process(command, numargs, commandargs, sim, traf, scr, cmd):
    #First, find by which callsign the CFreeFlight module is called in Cstack.py
    for sign,filename in cmd.extracmdmodules.iteritems():
        if filename==__name__:
            callsign = sign
    
    #change display settings and delete AC to generate own FF scenarios
    if command == "ON":
        if numargs == 0:
            traf.Mylog.texpstart=0
        else:
            traf.Mylog.texpstart=float(commandargs[1])*60
            
    elif command == "OFF":
        if numargs == 0:
            traf.Mylog.texpstop=0
        else:
            traf.Mylog.texpstop=float(commandargs[1])*60
    
    elif command == "SAVE":
        traf.Mylog.saveall()
    
    elif command == "CONF":
        if numargs == 0:
            scr.echo("CONF ON/OFF")
            if traf.Mylog.confsave:
                scr.echo("CONF are currently logged")
            else:
                scr.echo("CONF are currently not logged")
        else:
            arg1 = commandargs[1]  # arguments are strings
            traf.Mylog.confsave = (arg1.upper() =="ON")
    
    elif command == "SNAP":
        if numargs == 0:
            scr.echo("SNAP [ON/OFF] [TIME]")
            if traf.Mylog.snapsave:
                scr.echo("SNAP are currently logged")
            else:
                scr.echo("SNAP are currently not logged")
        else:
            arg1 = commandargs[1]  # arguments are strings
            traf.Mylog.snapsave = (arg1.upper() =="ON")
            if numargs==2:
                traf.Mylog.logperiod=float(commandargs[2])
                
    elif command == "FLST":
        if numargs == 0:
            scr.echo("FLST ON/OFF")
            if traf.Mylog.flstsave:
                scr.echo("FLST are currently logged")
            else:
                scr.echo("FLST are currently not logged")
        else:
            arg1 = commandargs[1]  # arguments are strings
            traf.Mylog.flstsave = (arg1.upper() =="ON")

    elif command == "SUM":
        if numargs == 0:
            scr.echo("SUM ON/OFF")
            if traf.Mylog.compsave:
                scr.echo("SUM are currently logged")
            else:
                scr.echo("SUM are currently not logged")
        else:
            arg1 = commandargs[1]  # arguments are strings
            traf.Mylog.compsave = (arg1.upper() =="ON")
    
    else:
        scr.echo("Unknown command: " + callsign + command)
    pass

