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
            if traf.dbconf.swasas:
                scr.echo("ASAS is currently ON")
            else:
                scr.echo("ASAS is currently OFF")
        else:
            arg1 = commandargs[1]  # arguments are strings
            traf.dbconf.swasas = (arg1.upper() =="ON")
    
    elif command == "RESO":
        if numargs== 0:
            scr.echo("RESO TYPE")
        else:
            if commandargs[1]=="OFF":
                traf.dbconf.SetCRmethod("DoNothing")
            elif commandargs[1]=="MVP":
                traf.dbconf.SetCRmethod("MVP")
            elif commandargs[1]=="SWARM":
                traf.dbconf.SetCRmethod("Swarm")
            elif commandargs[1]=="DIFGAME":
                traf.dbconf.SetCRmethod("Difgame")
            elif commandargs[1]=="EBY":
                traf.dbconf.SetCRmethod("Eby")
            elif commandargs[1]=="MVP_LAY":
                traf.dbconf.SetCRmethod("MVP_LAY")
            else:
                scr.echo("RESO TYPE UNKNOWN")

    elif command == "ZONER":
        if numargs ==0:
            scr.echo("ZONER <DIST> (NM)")
            scr.echo("CURRENT ZONER: "+str(traf.dbconf.R/nm)+" NM")
        else:
            traf.dbconf.R=float(commandargs[1])*nm
            traf.dbconf.Rm=np.maximum(traf.dbconf.R,traf.dbconf.Rm)

    elif command == "ZONEDH":
        if numargs ==0:
            scr.echo("ZONEDH <DIST> (FT)")
            scr.echo("CURRENT ZONEDH: "+str(traf.dbconf.dh/ft)+" FT")
        else:
            traf.dbconf.dh=float(commandargs[1])*ft
            traf.dbconf.dhm=np.maximum(traf.dbconf.dh,traf.dbconf.dhm)

    elif command == "RSZONER":
        if numargs ==0:
            scr.echo("RSZONER <DIST> (NM)")
            scr.echo("CURRENT RSZONER: "+str(traf.dbconf.Rm/nm)+" NM")
        else:
            if float(commandargs[1])*nm<traf.dbconf.R:
                scr.echo("RSZONER MAY NOT BE SMALLER THAN ZONER")
            else:
                traf.dbconf.Rm=float(commandargs[1])*nm

    elif command == "RSZONEDH":
        if numargs ==0:
            scr.echo("RSZONEDH <DIST> (FT)")
            scr.echo("CURRENT RSZONEDH: "+str(traf.dbconf.dhm/ft)+" FT")
        else:
            if float(commandargs[1])*ft<traf.dbconf.dh:
                scr.echo("RSZONEDH MAY NOT BE SMALLER THAN ZONEDH")
            else:
                traf.dbconf.dhm=float(commandargs[1])*ft

    elif command == "DTLOOK":
        if numargs ==0:
            scr.echo("DTLOOK <TIME>")
            scr.echo("CURRENT DTLOOK: "+str(traf.dbconf.dtlookahead)+" SEC")
        else:
            print commandargs
            traf.dbconf.dtlookahead=float(commandargs[1])
            

    elif command == "DTNOLOOK":
        if numargs ==0:
            scr.echo("DTNOLOOK <TIME>")
            scr.echo("CURRENT DTNOLOOK: "+str(traf.dtasas)+" SEC")
        else:
            traf.dtasas=float(commandargs[1])
    
    elif command == "DIR":
        if numargs ==0:
            scr.echo("DIR direction(COMB or HORIZ or VERT)")
            scr.echo("CURRENT DIR: " + traf.dbconf.swresodir )
        elif commandargs[1] == "COMB" or commandargs[1] == "HORIZ" or commandargs[1] == "VERT":
            traf.dbconf.SetResoDirection(commandargs[1])
        else:
            scr.echo("DIR is unknown. Try again!")
            
    elif command == "PRIO":
        if numargs ==0:
            scr.echo("PRIO [ON/OFF]")
            scr.echo("CURRENT PRIO: " + traf.dbconf.swprio )
        elif commandargs[1] == "ON": 
            traf.dbconf.swprio = True
        elif commandargs[1] == "OFF" or commandargs[1] == "OF": 
            traf.dbconf.swprio = False
        else:
            scr.echo("PRIO is unknown. Try again!")        

    elif command == "DELAY":
        if numargs ==0:
            scr.echo("DELAY [ON/OFF]")
        elif commandargs[1] == "ON":
            traf.dbconf.swdelay = True
            scr.echo("DELAY ON")
        elif commandargs[1] == "OFF" or commandargs[1] == "OF":
            traf.dbconf.swdelay = False
            scr.echo("DELAY OFF")
        else:
            scr.echo("PRIO is unknown. Try again!")
    else:
        scr.echo("Unknown command: " + callsign + command)

