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
    if command == "ASAS":
        if numargs == 0:
            scr.echo("ASAS ON/OFF")
            if traf.asas.swasas:
                scr.echo("ASAS is currently ON")
            else:
                scr.echo("ASAS is currently OFF")
        else:
            arg1 = commandargs[1]  # arguments are strings
            traf.asas.swasas = (arg1.upper() =="ON")
    
    elif command == "RESO":
        if numargs== 0:
            scr.echo("RESO TYPE")
        else:
            if commandargs[1]=="OFF":
                traf.asas.SetCRmethod("DoNothing")
            elif commandargs[1]=="MVP":
                traf.asas.SetCRmethod("MVP")
            elif commandargs[1]=="SWARM":
                traf.asas.SetCRmethod("Swarm")
            elif commandargs[1]=="DIFGAME":
                traf.asas.SetCRmethod("Difgame")
            elif commandargs[1]=="EBY":
                traf.asas.SetCRmethod("Eby")
            else:
                scr.echo("RESO TYPE UNKNOWN")

    elif command == "ZONER":
        if numargs ==0:
            scr.echo("ZONER <DIST> (NM)")
            scr.echo("CURRENT ZONER: "+str(traf.asas.R/nm)+" NM")
        else:
            traf.asas.R=float(commandargs[1])*nm
            traf.asas.Rm=np.maximum(traf.asas.R,traf.asas.Rm)

    elif command == "ZONEDH":
        if numargs ==0:
            scr.echo("ZONEDH <DIST> (FT)")
            scr.echo("CURRENT ZONEDH: "+str(traf.asas.dh/ft)+" FT")
        else:
            traf.asas.dh=float(commandargs[1])*ft
            traf.asas.dhm=np.maximum(traf.asas.dh,traf.asas.dhm)

    elif command == "RSZONER":
        if numargs ==0:
            scr.echo("RSZONER <DIST> (NM)")
            scr.echo("CURRENT RSZONER: "+str(traf.asas.Rm/nm)+" NM")
        else:
            if float(commandargs[1])*nm<traf.asas.R:
                scr.echo("RSZONER MAY NOT BE SMALLER THAN ZONER")
            else:
                traf.asas.Rm=float(commandargs[1])*nm

    elif command == "RSZONEDH":
        if numargs ==0:
            scr.echo("RSZONEDH <DIST> (FT)")
            scr.echo("CURRENT RSZONEDH: "+str(traf.asas.dhm/ft)+" FT")
        else:
            if float(commandargs[1])*ft<traf.asas.dh:
                scr.echo("RSZONEDH MAY NOT BE SMALLER THAN ZONEDH")
            else:
                traf.asas.dhm=float(commandargs[1])*ft

    elif command == "DTLOOK":
        if numargs ==0:
            scr.echo("DTLOOK <TIME>")
            scr.echo("CURRENT DTLOOK: "+str(traf.asas.dtlookahead)+" SEC")
        else:
            print commandargs
            traf.asas.dtlookahead=float(commandargs[1])
            

    elif command == "DTNOLOOK":
        if numargs ==0:
            scr.echo("DTNOLOOK <TIME>")
            scr.echo("CURRENT DTNOLOOK: "+str(traf.dtasas)+" SEC")
        else:
            traf.dtasas=float(commandargs[1])

    else:
        scr.echo("Unknown command: " + callsign + command)

