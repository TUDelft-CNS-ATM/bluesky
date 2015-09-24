from math import *
import numpy as np
from random import random, randint
import os

from ..tools.aero import kts, ft, fpm, nm, lbs, \
    qdrdist, cas2tas, mach2tas, tas2cas, tas2eas, tas2mach, eas2tas, cas2mach, density
from ..tools.misc import txt2alt, txt2spd, col2rgb, cmdsplit
from .. import settings

class Commandstack:
    """
    Commandstack class definition : command stack & processing class

    Methods:
        Commandstack()          :  constructor

        stack(cmdline)          : add a command to the command stack
        openfile(scenname)      : start playing a scenario file scenname.SCN
                                  from scenario folder
        savefile(scenname,traf) : save current traffic situation as
                                  scenario file scenname.SCN
        checkfile(t)            : check whether commands need to be
                                  processed from scenario file

        process(sim, traf, scr) : central command processing method
                                  (with long elif tree with all commands)

    Created by  : Jacco M. Hoekstra (TU Delft)
    """
    def __init__(self):

        self.cmdstack  = []
        self.scenlines = []
        self.scentime  = []
        self.scenfile  = ""
        self.scentime = []
        self.scencmd = []

        # An advanced way to add your own commands: add your entry to the dictionary.
        # The dictionary should be formed as {"Key":'module'}.
        # "Key" is a TWO-letterd reference that is used at the start of the command.
        # 'module' is the name of the .py-file in which the commands are located (without .py).
        # Make sure that the module has a function "process" with
        # arguments (command, number of args, array of args, sim, traf, scr, cmd)
        # self.extracmdmodules={"FF_": "freeflight"}
        self.extracmdmodules = {}

        # Import modules from the list
        self.extracmdrefs = {}
        for key in self.extracmdmodules:
            obj = __import__(self.extracmdmodules[key], globals(), locals(), [], 0)
            self.extracmdrefs[key] = obj

        return

    def stack(self, cmdline):  # Stack one or more commands separated by ";"
        cline = cmdline.strip()  # remove leading & trailing spaces
        if cline.count(";") == 0:
            self.cmdstack.append(cline)
        else:
            clinelst = cline.split(";")
            for line in clinelst:
                self.cmdstack.append(line)

        return

    def openfile(self, scenname):
        # No filename: empty start
        self.scenlines = []

        # Add .scn extension if necessary
        if scenname.lower().find(".scn") < 0:
            scenname = scenname + ".scn"

        # If it is with a path don't touch it, else add path
        if scenname.find("/") < 0:
            scenfile = settings.scenario_path
            if scenfile[-1] is not '/':
                scenfile += '/'
            scenfile += scenname.lower()
        else:
            scenfile = scenname

        print "Reading scenario file: ", scenfile
        print

        # Read lines into buffer
        self.scentime = []
        self.scencmd = []

        if os.path.exists(scenfile):
            fscen = open(scenfile, 'r')
            self.scenlines = fscen.readlines()
            fscen.close()
            i = 0
            while i < len(self.scenlines):
                if len(self.scenlines[i].strip()) <= 12 or \
                                self.scenlines[i][0] == "#":

                    del self.scenlines[i]
                else:
                    i = i + 1

            # Optional?
            # scenlines.sort()

            # Set timer until what is read
            # tstamp = self.scenlines[0][:11]    # format - hh:mm:ss.hh
            # ihr = int(tstamp[:2])
            # imin = int(tstamp[3:5])
            # isec = float(tstamp[6:8]+"."+tstamp[9:11])

            # Split scenario file line in times and commands
            for line in self.scenlines:
                # lstrip = line.strip()
                # Try reading timestamp and command
                try:
                    icmdline = line.index('>')
                    tstamp = line[:icmdline]
                    ttxt = tstamp.strip().split(':')
                    ihr = int(ttxt[0])
                    imin = int(ttxt[1])
                    xsec = float(ttxt[2])
                    self.scentime.append(ihr * 3600. + imin * 60. + xsec)
                    self.scencmd.append(line[icmdline + 1:-1])
                except:
                    print "except this:", line
                    pass  # nice try, we will just ignore this syntax error

        else:
            print"Error: cannot find file:", scenfile

        return

    def checkfile(self, simt):
        # Empty command buffer when it's time
        while len(self.scencmd) > 0 and simt >= self.scentime[0]:
            self.stack(self.scencmd[0])
            del self.scencmd[0]
            del self.scentime[0]

        return

    def saveic(self, fname, sim, traf):

        # Add extension .scn if not already present
        if fname.find(".scn") < 0 and fname.find(".SCN"):
            fname = fname + ".scn"

        # If it is with path don't touch it, else add path
        if fname.find("/") < 0:
            scenfile = "./scenario/" + fname.lower()

        try:
            f = open(scenfile, "w")
        except:
            return -1

        # Write files
        timtxt = "00:00:00.00>"

        for i in range(traf.ntraf):
            # CRE acid,type,lat,lon,hdg,alt,spd
            cmdline = "CRE " + traf.id[i] + "," + traf.type[i] + "," + \
                      repr(traf.lat[i]) + "," + repr(traf.lon[i]) + "," + \
                      repr(traf.trk[i]) + "," + repr(traf.alt[i] / ft) + "," + \
                      repr(tas2cas(traf.tas[i], traf.alt[i]) / kts)

            f.write(timtxt + cmdline + chr(13) + chr(10))

            # VS acid,vs
            if abs(traf.vs[i]) > 0.05:  # 10 fpm dead band
                if abs(traf.avs[i]) > 0.05:
                    vs_ = traf.avs[i] / fpm
                else:
                    vs_ = traf.vs[i] / fpm

                cmdline = "VS " + traf.id[i] + "," + repr(vs_)
                f.write(timtxt + cmdline + chr(13) + chr(10))

            # Autopilot commands
            # Altitude
            if abs(traf.alt[i] - traf.aalt[i]) > 10.:
                cmdline = "ALT " + traf.id[i] + "," + repr(traf.aalt[i] / ft)
                f.write(timtxt + cmdline + chr(13) + chr(10))

            # Heading as well when heading select
            delhdg = (traf.trk[i] - traf.ahdg[i] + 180.) % 360. - 180.
            if abs(delhdg) > 0.5:
                cmdline = "HDG " + traf.id[i] + "," + repr(traf.ahdg[i])
                f.write(timtxt + cmdline + chr(13) + chr(10))

            # Speed select? => Record
            rho = density(traf.alt[i])  # alt in m!
            aptas = sqrt(1.225 / rho) * traf.aspd[i]
            delspd = aptas - traf.tas[i]

            if abs(delspd) > 0.4:
                cmdline = "SPD " + traf.id[i] + "," + repr(traf.aspd[i] / kts)
                f.write(timtxt + cmdline + chr(13) + chr(10))

            # DEST acid,dest-apt
            if traf.dest[i] != "":
                cmdline = "DEST " + traf.id[i] + "," + traf.dest[i]
                f.write(timtxt + cmdline + chr(13) + chr(10))

            # ORIG acid,orig-apt
            if traf.orig[i] != "":
                cmdline = "ORIG " + traf.id[i] + "," + \
                          traf.orig[i]
                f.write(timtxt + cmdline + chr(13) + chr(10))

        # Saveic: should close
        f.close()
        return 0

    def process(self, sim, traf, scr):
        """process and empty command stack"""

        # Process stack of commands
        for line in self.cmdstack:
            cmdline = line.upper()  # Save original lower case in variable line

            cmdargs = cmdsplit(cmdline)

            # Empty line: next command
            if len(cmdargs) == 0 or cmdline.strip() == "":
                continue

            cmd = cmdargs[0]

            numargs = len(cmdargs) - 1

            # First check for alternate syntax: acid cmd args 2-3
            if cmd != "" and traf.id.count(cmd) > 0:
                if numargs >= 1:
                    acid = cmd
                    cmd = cmdargs[1]
                    cmdargs[1] = acid
                    cmdargs[0] = cmd
                else:
                    cmdargs.append(cmdargs[0])
                    cmdargs[0] = 'POS'
                    cmd = 'POS'
                    numargs = 1

            # Assume syntax is ok (default)
            synerr = False

            # Catch general errors
            if True:  # optional to switch error protection off

                #**********************************************************************
                #=====================  Start of command branches =====================
                #**********************************************************************

                #----------------------------------------------------------------------
                # QUIT/STOP/END/Q: stop program
                #----------------------------------------------------------------------
                if cmd == "QUIT" or cmd == "STOP" or cmd == "END" \
                        or cmd == "EXIT" or cmd[0] == "Q":

                    sim.stop()

                #----------------------------------------------------------------------
                # HELP/?: HELP command
                #----------------------------------------------------------------------
                elif cmd[:4] == "HELP" or cmd=="?" and numargs==0:
                    scr.echo("To get help on a command,"+\
                             " enter it without arguments."+\
                             "Some basic commands are given below:")
                    scr.echo(" ")
                    scr.echo("CRE HDG SPD ALT DEL OP HOLD QUIT DEST ORIG")
                    scr.echo("MCRE ADDWPT DELWPT LISTRTE LNAV VNAV")
                    scr.echo("POS ZOOM PAN SWRAD AREA")
                    scr.echo("DATAFEED")
                    scr.echo(" ")
                    scr.echo("See Doc folder for more info.")


                #----------------------------------------------------------------------
                # CRE: CREate command: Create an aircraft
                #----------------------------------------------------------------------
                elif cmd[:3] == "CRE":
                    if numargs <= 1:
                        scr.echo("CRE acid,type,lat,lon,hdg,alt,spd")
                        continue  # next command
                    elif numargs >= 2:
                        acid = cmdargs[1].upper()  # arguments are strings
                        actype = cmdargs[2]  # arguments are strings
                        if traf.id.count(acid) > 0:
                            scr.echo("CRE error: " + acid + " already exists.")
                            continue
                    
                    # Decode with check
                    try:
                        aclat = float(cmdargs[3])  # deg
                        aclon = float(cmdargs[4])  # deg
                        achdg = float(cmdargs[5])  # deg
                        acalt = txt2alt(cmdargs[6]) * ft  # m

                        if acalt <= -900:
                            synerr = True
                            acalt = 0.

                        # Speed
                        acspd = txt2spd(cmdargs[7], acalt)  # kts/M => m/s
                        if acspd < 0.:
                            synerr = True
                    except:
                        synerr = True

                    if not synerr:
                        traf.create(acid, actype, aclat, aclon, achdg, acalt, acspd)
                        i = traf.id2idx(acid)
                        traf.crzalt[i] = acalt  # Use CRE-alt as default crzalt
                    else:
                        print "Syntax error in command"
                        scr.echo("Syntax error in command")
                        scr.echo("CRE acid,type,lat,lon,hdg,alt,spd")
                        continue

                #----------------------------------------------------------------------
                # POS command: traffic info;
                #----------------------------------------------------------------------
                elif cmd == "POS" or cmd=="?":
                    if numargs >= 1:
                        acid = cmdargs[1]

                        # Does aircraft exist?
                        idx = traf.id2idx(acid)
                        if idx < 0:
                            scr.echo("POS: " + acid + " not found.")

                        # print info on aircraft if found
                        else:
                            scr.echo("Info on " + acid + " " + traf.type[idx])
                            scr.echo("Pos = " + str((traf.lat[idx], traf.lon[idx])))
                            scr.echo(str(int(traf.tas[idx] / kts)) + " kts at " \
                                     + str(int(traf.alt[idx] / ft)) + " ft")
                            scr.echo("Hdg = " + str(int(traf.trk[idx])))
                            if traf.swvnav[idx]:
                                vnavtxt = "VNAV "
                            else:
                                vnavtxt = ""
                            if traf.swlnav[idx] and traf.route[idx].nwp>0 and  \
                               traf.route[idx].iactwp>=0:
                                 scr.echo(vnavtxt + "LNAV to "+   \
                                  traf.route[idx].wpname[traf.route[idx].iactwp])
                            
                            txt = "Flying"
                            if traf.orig[idx]!="":
                                txt = txt + " from "+traf.orig[idx]
                            if traf.dest[idx]!="":
                                txt = txt + " to "+traf.dest[idx]
                            if len(txt)>0:
                                scr.echo(txt)

                        # Show route for this aircraft (toggle switch)
                        scr.showroute(acid)

                    else:
                         synerr = True

                #----------------------------------------------------------------------
                # MOVE command: Move aircraft: MOVE acid, lat,lon[,alt,hdg,spd,vspd]
                #----------------------------------------------------------------------
                elif cmd == "MOVE":
                    if numargs >= 1:
                        acid = cmdargs[1]

                        # Does aircraft exist?
                        idx = traf.id2idx(acid)
                        if idx < 0:
                            scr.echo("MOVE: " + acid + " not found.")
                            idx = -1

                        # Move aircraft for which there are data
                        if idx >= 0:

                            if numargs >= 3:  # Position lat,lon
                                if cmdargs[2] != "":
                                    lat = float(cmdargs[2])
                                    traf.lat[idx] = lat
                                if cmdargs[3] != "":
                                    lon = float(cmdargs[3])
                                    traf.lon[idx] = lon

                            if numargs >= 4 and cmdargs[4] != "":  # altitude
                                alt = txt2alt(cmdargs[4])
                                traf.alt[idx] = alt * ft
                                traf.aalt[idx] = alt * ft

                            if numargs >= 5 and cmdargs[5] != "":  # heading(track)
                                traf.trk[idx] = float(cmdargs[5])
                                traf.ahdg[idx] = traf.trk[idx]

                            if numargs >= 6 and cmdargs[6] != "":  # speed
                                acspd = txt2spd(cmdargs[6], traf.alt[idx])

                                if acspd > 0.:
                                    traf.tas[idx] = acspd
                                    traf.aspd[idx] = tas2eas(traf.tas[idx], traf.alt[idx])
                                else:
                                    syntax = False

                            if numargs >= 7 and cmdargs[7] != "":  # vertical speed
                                traf.vs[idx] = float(cmdargs[7]) * fpm

                    else:
                        scr.echo("MOVE acid,lat,lon,[alt,hdg,spd,vspd]")

                #----------------------------------------------------------------------
                # DEL: Delete command: delete an aircraft
                #----------------------------------------------------------------------
                elif cmd == "DEL":
                    if numargs == 1:
                        if cmdargs[1]=="LINE" or cmdargs[1]=="LINES":
                            scr.objdel() # delete raw objects
                        else:
                            success = traf.delete(cmdargs[1].upper())
                            if not success:
                               sim.datalog.write(sim.t,
                                          "DEL: " + cmdargs[1] + " not found.")

                #----------------------------------------------------------------------
                # ALT command: altitude [ft/FLnnn], [VS [fpm]] 
                # altitude autopilot command
                #----------------------------------------------------------------------
                elif cmd == "ALT":
                    if numargs < 2:
                        scr.echo("ALT acid,alt[ft]/FLnnn")
                    elif numargs >= 2:
                        acid = cmdargs[1].upper()
                        idx = traf.id2idx(acid)
                        if idx >= 0:
                            try:
                                # update flight level as well as altitude
                                traf.aalt[idx] = txt2alt(cmdargs[2]) * ft
                                traf.afll[idx] = (traf.aalt[idx])/(100*ft)
                                traf.swvnav[idx] = False

                                delalt = traf.aalt[idx] - traf.alt[idx]

                                # Check for VS with opposite sign => use default vs by setting autopilot vs to zero
                                if traf.avs[idx] * delalt < 0. and abs(traf.avs[idx]) > 0.01:
                                    traf.avs[idx] = 0.

                                # Check for optional VS argument
                                if numargs == 3:
                                    traf.avs[idx] = float(cmdargs[3]) * fpm

                            except:
                                syntax = False

                        else:
                            scr.echo(cmd + ": " + acid + " not found.")

                    else:
                        syntax = False

                #----------------------------------------------------------------------
                # HDG heading [deg,true] heading autpilot command
                #----------------------------------------------------------------------
                elif cmd == "HDG":
                    if numargs < 2:
                        scr.echo("HDG acid,hdg[deg,True]")
                    elif numargs == 2:
                        acid = cmdargs[1].upper()
                        idx = traf.id.index(acid)
                        if idx >= 0:
                            traf.ahdg[idx] = float(cmdargs[2])
                            traf.swlnav[idx] = False
                        else:
                            scr.echo(cmd + ": " + acid + " not found.")
                    else:
                        syntax = False

                #----------------------------------------------------------------------
                # SPD speed [kts, CAS] Speed autopilot command
                # TODO: add Mach, and note how currently EAS is stored not CAS!
                #----------------------------------------------------------------------
                elif cmd == "SPD":
                    if numargs < 2:
                        scr.echo("SPD acid,spd[kts,EAS]")
                    elif numargs == 2:
                        acid = cmdargs[1].upper()
                        idx = traf.id2idx(acid)
                        if idx >= 0:
                            # cascmd = float(cmdargs[2])*kts
                            # tascmd = cas2tas(cascmd,traf.alt[idx])
                            # eascmd = tas2eas(tascmd,traf.alt[idx])

                            eascmd = float(cmdargs[2]) * kts
                            traf.aspd[idx] = eascmd
                            traf.ama[idx] = cas2mach(eascmd, traf.alt[idx])
                            traf.swvnav[idx] = False  # switch off VNAV

                        else:
                            scr.echo(cmd + ": " + acid + " not found.")
                    else:
                        syntax = False
              
                #----------------------------------------------------------------------
                # reset acceleration back to nominal value 
                #      of 1 kt/s^2 respectively 0.514444 m/s^2
                #----------------------------------------------------------------------
                elif cmd == "NOM":
                    if numargs < 1:
                        scr.echo("NOM acid, NOM")
                    elif numargs == 1:
                        acid = cmdargs[1].upper()
                        idx = traf.id2idx(acid)
                        if idx >= 0:
                            traf.ax[idx] = kts
                        else:
                            scr.echo(cmd + ": " + acid + " not found.")
                    else:
                        syntax = False                

                #----------------------------------------------------------------------
                # VS vertspeed [ft/min] Vertical speed autopilot command
                #----------------------------------------------------------------------
                elif cmd == "VS":
                    if numargs < 2:
                        scr.echo("VS acid,vspd [ft/min]")
                    elif numargs == 2:
                        acid = cmdargs[1]
                        idx = traf.id2idx(acid)
                        if idx >= 0:
                            traf.avs[idx] = float(cmdargs[2]) * fpm
                            # traf.vs[idx]=float(cmdargs[2]) * fpm
                            traf.swvnav[idx] = False
                        else:
                            scr.echo(cmd + ": " + acid + " not found.")
                    else:
                        syntax = False

                #----------------------------------------------------------------------
                # DEST/ORIG: Destination/Origin command: set destination/origin airport
                #----------------------------------------------------------------------
                elif cmd == "DEST" or cmd == "ORIG":
                    if numargs==0:
                        scr.echo(cmd+" acid, airport")
                    elif numargs >=1:
                        acid = cmdargs[1]
                        idx = traf.id2idx(acid)
                        if idx>=0 and numargs==1:
                            if cmd=="DEST":
                                scr.echo("DEST "+acid+": "+traf.dest[idx])                            
                            else:
                                scr.echo("ORIG "+acid+": "+traf.orig[idx])                            
                            
                        elif idx >= 0 and numargs>=2:
                            # Destination is default waypoint
                            if cmd == "DEST":
                                traf.dest[idx] = cmdargs[2].upper().strip()
                                iwp = traf.route[idx].addwpt(traf,i,traf.dest[idx],
                                                               traf.route[idx].dest,
                                                               traf.lat[idx], traf.lon[idx],
                                                               0.0, traf.cas[idx])
                                # If only waypoint: activate
                                if (iwp == 0) or (traf.orig[idx]!="" and traf.route[idx].nwp==2):
                                     traf.actwplat[idx] = traf.route[idx].wplat[iwp]
                                     traf.actwplon[idx] = traf.route[idx].wplon[iwp]
                                     traf.actwpalt[idx] = traf.route[idx].wpalt[iwp]
                                     traf.actwpspd[idx] = traf.route[idx].wpspd[iwp]

                                     traf.swlnav[idx] = True
                                     traf.route[idx].iactwp = iwp

                                # If not found, say so
                                elif iwp < 0:
                                     scr.echo(traf.dest[idx] + " not found.")

                            # Origin: bookkeeping only for now
                            else:
                                traf.orig[idx] = cmdargs[2]
                                iwp = traf.route[idx].addwpt(traf,i,traf.orig[idx],traf.route[idx].orig,
                                                 traf.lat[idx], traf.lon[idx],
                                                          0.0, traf.cas[idx])
                                if iwp < 0:
                                     scr.echo(traf.orig[idx] + " not found.")
                        # Give destination and origin
                        elif idx >=0 and numargs==1:
                            txt = ""
                            if traf.orig[idx]!="":
                                txt = txt + " from "+traf.orig[idx]
                            if traf.dest[idx]!="":
                                txt = txt + " to "+traf.dest[idx]
                            if len(txt)>0:
                                scr.echo(traf.id[idx]+txt)
                            else:
                                scr.echo(traf.id[idx]+": orig and dest not defined.")

                        else:
                            scr.echo(cmd + ": aircraft " + acid + " not found.")
                    else:
                        syntax = False

                #----------------------------------------------------------------------
                # ZOOM command (or use ++++  or --  to zoom in or out)
                #----------------------------------------------------------------------
                elif cmd[:4] == "ZOOM" or cmd[0] == "+" or cmd[0] == "=" or cmd[0] == "-":
                    if cmd[0] != "Z":
                        nplus = cmd.count("+") + cmd.count("=")  #= equals + (same key)
                        nmin = cmd.count("-")
                        zoomfac = sqrt(2) ** nplus / (sqrt(2) ** nmin)
                        scr.zoom(zoomfac)
                    else:
                        syntax = len(cmdargs) == 2
                        if syntax:
                            if cmdargs[1] == "IN":
                                scr.zoom(1.4142135623730951)  # sqrt(2.)

                            elif cmdargs[1] == "OUT":
                                scr.zoom(0.70710678118654746)  #1./sqrt(2.)
                            else:
                                syntax = False

                        if not syntax:
                            print "Syntax error in command"
                            scr.echo("Syntax error in command")
                            scr.echo("ZOOM IN/OUT")
                            continue  # Skip default syntyax message

                #----------------------------------------------------------------------
                # PAN command
                #----------------------------------------------------------------------
                elif cmd[:4] == "PAN":

                    if not (numargs == 1 or numargs == 2):
                        scr.echo("Syntax error in command")
                        scr.echo("PAN LEFT/RIGHT/UP/DOWN/acid/airport/navid")
                        continue

                    # LEFT/RIGHT/UP/DOWN
                    elif numargs == 1:
                        if cmdargs[1] == "LEFT":
                            scr.pan((0.0, -0.5)) # move half screen left
                        elif cmdargs[1] == "RIGHT":
                            scr.pan((0.0, 0.5)) # move half screen right

                        elif cmdargs[1] == "UP":
                            scr.pan((0.5, 0.0))  # move half screen up

                        elif cmdargs[1] == "DOWN":
                            scr.pan((-0.5, 0.0)) # move half screen down
                        else:

                            # Try aicraft id, waypoint of airport
                            i = traf.id2idx(cmdargs[1])
                            if i >= 0:
                                lat = traf.lat[i]
                                lon = traf.lon[i]
                                if (np.isnan(lat) or np.isnan(lon)):
                                    continue
                            else:
                                i = traf.navdb.getwpidx(cmdargs[1], 0.0, 0.0)  # TODO: get current pan from display?
                                if i >= 0:
                                    lat = traf.navdb.wplat[i]
                                    lon = traf.navdb.wplon[i]
                                    if (np.isnan(lat) or np.isnan(lon)):
                                        continue
                                else:
                                    i = traf.navdb.getapidx(cmdargs[1])
                                    if i >= 0:
                                        lat = traf.navdb.aplat[i]
                                        lon = traf.navdb.aplon[i]
                                        if (np.isnan(lat) or np.isnan(lon)):
                                            continue
                                    else:
                                        scr.echo(cmdargs[1] + " not found.")

                            # PAN to lat,lon position
                            if numargs == 2:
                                lat = float(cmdargs[1])
                                lon = float(cmdargs[2])

                            if not (np.isnan(lat) or np.isnan(lon)):
                                scr.pan((lat, lon), absolute=True)

                #----------------------------------------------------------------------
                # NAVDISP/ND  acid:  Activate Navdisplay mode
                #----------------------------------------------------------------------
#                elif cmd == "ND" or cmd == "NAVDISP":
#
#                    if numargs < 1:  # Help text
#                        scr.echo("NAVDISP acid/OFF")
#                        if scr.swnavdisp:
#                            scr.echo("Ownship is" + scr.ndacid)
#                        else:
#                            scr.echo("NAVDISP is off")
#
#                    # Or switch off
#                    elif cmdargs[1] == "OFF":
#                        scr.swnavdisp = False
#                        scr.redrawradbg = True
#                        scr.geosel = ()
#                        scr.firsel = ()
#
#                    # Follow aircraft
#                    else:
#                        i = traf.id2idx(cmdargs[1])
#                        if i >= 0:
#                            scr.ndacid = cmdargs[1]
#                            scr.swnavdisp = True
#                            scr.redrawradbg = True
#                            scr.geosel = ()
#                            scr.firsel = ()
#                        else:
#                            scr.echo("NAVDISP: " + cmdargs[1] + " not found.")
#
                #----------------------------------------------------------------------
                # RUNFT: toggle fast time running
                #----------------------------------------------------------------------
                elif cmd == "RUNFT":
                    sim.run_fixed = (not sim.run_fixed)

                #----------------------------------------------------------------------
                # IC scenfile: command: restart with new filename (.scn will be added if necessary)
                # IC IC: same file
                #----------------------------------------------------------------------
                elif cmd == "IC":
                    # If no arg is given: check
                    if numargs >= 1:
                        # Use lower case line for filename and allow space in path
                        filename = line.strip()[3:].strip()
                        if filename.upper() == "IC":  # same file
                            filename = self.scenfile

                        if filename.strip() != "":
                            scr.echo("Opening " + filename + " ...")

                        # Open file in ./scenario subfolder
                        sim.reset()
                        traf.deleteall()

                        self.scenfile = filename
                        self.openfile(self.scenfile)
                    else:
                        filename = scr.show_file_dialog()
                        if len(filename) > 0:
                            self.scenfile = filename
                            self.openfile(self.scenfile)


                #----------------------------------------------------------------------
                # OP: Continue to run
                #----------------------------------------------------------------------
                elif cmd == "OP" or cmd == "START" or cmd == "CONTINUE" or cmd == "RUN":
                    sim.start()

                #----------------------------------------------------------------------
                # HOLD/PAUSE: HOLD/PAUSE mode
                #----------------------------------------------------------------------
                elif cmd == "HOLD" or cmd == "PAUSE":
                    sim.pause()

                #----------------------------------------------------------------------
                # SAVE/SAVEIC Save current traffic situation as IC scn file
                #----------------------------------------------------------------------
                elif cmd == "SAVEIC":
                    if numargs <= 0:
                        scr.echo("SAVEIC needs filename")
                    else:
                        errcode = self.saveic(cmdargs[1], sim, traf)
                        if errcode == -1:
                            scr.echo("SAVEIC: Error writing file")

                #----------------------------------------------------------------------
                # DT: set value of DT for FIXDT mode
                #----------------------------------------------------------------------
                elif cmd == "DT":
                    if numargs < 1:
                        scr.echo("DT [dt] sets DT for fixdt mode")
                        scr.echo("Current dt = " + str(sim.fixdt))
                    else:
                        dt_ = float(cmdargs[1])
                        sim.fixdt = abs(dt_)

                #----------------------------------------------------------------------
                # FIXDT: switch FIXDT mode on/off
                #----------------------------------------------------------------------
                elif cmd == "FIXDT":
                    if numargs < 1:
                        scr.echo("FIXDT ON/OFF [,howmanyseconds]")
                        scr.echo("Current dt = " + str(sim.fixdt))
                        if sim.ffmode:
                            scr.echo("FIXDT mode is ON")
                            if sim.ffstop > 0.:
                                t_ = sim.ffstop - sim.t
                                scr.echo("for " + str(t_) + " more seconds")
                        else:
                            scr.echo("FIXDT mode is OFF")
                    else:
                        if cmdargs[1].upper() == "ON":
                            sim.ffmode = True
                            if numargs >= 2:
                                try:
                                    tstop_ = float(cmdargs[2])
                                    sim.ffstop = abs(tstop_) + sim.t
                                except:
                                    sim.ffstop = -1.
                                    syntax = False  # syntax is not ok
                            else:
                                sim.ffmode = True
                                sim.ffstop = -1
 
                        elif cmdargs[1].upper()[:2] == "OF":
                            sim.ffmode = False

                #----------------------------------------------------------------------
                # METRICS command: METRICS/METRICS OFF/0/1/2 [dt]  analyze traffic complexity metrics
                #----------------------------------------------------------------------
                elif cmd[:6] == "METRIC":
                    if numargs < 1:
                        scr.echo("METRICS/METRICS OFF/0/1/2 [dt]")

                        if traf.metricSwitch == 1:
                            scr.echo("")
                            scr.echo("Active: " + "(" + str(traf.metric.metric_number + 1) + ") " + traf.metric.name[
                                traf.metric.metric_number])
                            scr.echo("Current dt: " + str(traf.metric.dt) + " s")
                        if traf.metricSwitch == 0:
                            scr.echo("No metric active")

                    elif cmdargs[1] == "ON":  # arguments are strings
                        scr.echo("METRICS/METRICS OFF/0/1/2 [dt]")

                    elif cmdargs[1] == "OFF":  # arguments are strings
                        traf.metricSwitch = 0
                        scr.echo("Metric is off")

                    else:
                        metric_number = int(cmdargs[1]) - 1
                        if metric_number == -1:
                            traf.metricSwitch = 0
                            scr.echo("Metric is off")

                        elif metric_number <= len(traf.metric.name) and metric_number >= 0:
                            if traf.area == "Circle":
                                traf.metricSwitch = 1
                                scr.echo("(" + str(traf.metric.metric_number + 1) + ") " + traf.metric.name[
                                    traf.metric.metric_number] + " activated")
                                try:
                                    metric_dt = float(cmdargs[2])
                                    if metric_dt > 0:
                                        traf.metric.dt = metric_dt
                                        scr.echo("with dt = " + str(metric_dt))
                                    else:
                                        scr.echo("No valid dt")
                                except:
                                    scr.echo("with dt = " + str(traf.metric.dt))
                            else:
                                scr.echo("First define AREA FIR")
                        else:
                            scr.echo("No such metric")


                #----------------------------------------------------------------------
                # AREA command: AREA lat0,lon0,lat1,lon1[,lowalt]
                #               AREA FIR fir radius [lowalt]
                #----------------------------------------------------------------------
                elif cmd == "AREA":

                    if numargs == 0:
                        scr.echo("AREA lat0,lon0,lat1,lon1[,lowalt]")
                        scr.echo("or")
                        scr.echo("AREA fir,radius[,lowalt]")
                    elif numargs == 1 and cmdargs[1] != "OFF" and cmdargs[1] != "FIR":
                        scr.echo("AREA lat0,lon0,lat1,lon1[,lowalt]")
                        scr.echo("or")
                        scr.echo("AREA fir,radius[,lowalt]")
                    elif numargs == 1:
                        if cmdargs[1] == "OFF":
                            traf.swarea = False
                            scr.drawradbg() # redraw radar background
                            traf.area = ""
                        if cmdargs[1] == "FIR":
                            scr.echo("Specify FIR")

                    elif numargs > 1 and cmdargs[1][0].isdigit():

                        lat0 = float(cmdargs[1])  # [deg]
                        lon0 = float(cmdargs[2])  # [deg]
                        lat1 = float(cmdargs[3])  # [deg]
                        lon1 = float(cmdargs[4])  # [deg]

                        traf.arealat0 = min(lat0, lat1)
                        traf.arealat1 = max(lat0, lat1)
                        traf.arealon0 = min(lon0, lon1)
                        traf.arealon1 = max(lon0, lon1)

                        if numargs == 5:
                            traf.areafloor = float(cmdargs[5]) * ft
                        else:
                            traf.areafloor = -9999999.

                        traf.area = "Square"
                        traf.swarea = True
                        scr.redrawradbg = True

                        # Avoid mass delete due to redefinition of area
                        traf.inside = traf.ntraf * [False]

                    elif numargs > 2 and cmdargs[1] == "FIR":

                        for i in range(0, len(traf.navdb.fir)):
                            if cmdargs[2] == traf.navdb.fir[i][0]:
                                break
                        traf.metric.fir_number = i
                        if cmdargs[2] != traf.navdb.fir[i][0]:
                            scr.echo("Uknown FIR, try again")
                        traf.metric.fir_circle_point = traf.metric.metric_Area.FIR_circle(traf.navdb, traf.metric.fir_number)
                        traf.metric.fir_circle_radius = float(cmdargs[3])

                        if numargs == 4:
                            traf.areafloor = float(cmdargs[4]) * ft
                        else:
                            traf.areafloor = -9999999.
                        if numargs > 4:
                            scr.echo("AREA command unknown")

                        traf.area = "Circle"
                        traf.swarea = True
                        scr.drawradbg()
                        traf.inside = traf.ntraf * [False]
                    else:
                        scr.echo("AREA command unknown")
                        scr.echo("AREA lat0,lon0,lat1,lon1[,lowalt]")
                        scr.echo("or")
                        scr.echo("AREA fir,radius[,lowalt]")

                
                #----------------------------------------------------------------------
                # TAXI command: TAXI ON/OFF : if off, 
                #   autodelete descending aircraft below 1500 ft
                #----------------------------------------------------------------------
                elif cmd == "TAXI":
                    if numargs == 0:
                        scr.echo("TAXI ON/OFF : OFF auto deletes traffic below 1500 ft")
                    else:
                        arg1 = cmdargs[1].upper()  # arguments are strings
                        traf.swtaxi = (arg1[:2] == "ON")

                #----------------------------------------------------------------------
                # SWRAD command: display switches of radar display
                # SWRAD GEO / GRID / APT / VOR / WPT / LABEL (toggle on/off or cycle)
                # (for WPT,APT,LABEL value is optional)
                #----------------------------------------------------------------------
                elif cmd == "SWRAD" or cmd[:4] == "DISP":
                    if numargs == 0:
                        scr.echo("SWRAD GEO / GRID / APT / VOR / " + \
                                 "WPT / LABEL / TRAIL [dt] / [value]")
                    else:
                        sw = cmdargs[1] # Which switch

                        if numargs==2:
                            arg = cmdargs[2] # optional argument
                        else:
                            arg = ""

                        # FIR boundaries
                        if sw == "TRAIL" or sw == "TRAILS":

                            traf.swtrails = not traf.swtrails
                            if numargs == 2:
                                try:
                                    trdt = float(cmdargs[2])
                                    traf.trails.dt = trdt
                                except:
                                    scr.echo("TRAIL ON dt")

                        elif scr.feature(sw,arg): # Toggle screen feature                        
                              scr.drawradbg() # When success: Force redraw radar background

                        else:
                            scr.redrawradbg = False # Switch not found

                #----------------------------------------------------------------------
                # TRAILS ON/OFF
                #----------------------------------------------------------------------
                elif cmd[:5] == "TRAIL":
                    if numargs == 0:
                        scr.echo("TRAIL ON/OFF [dt]/TRAIL acid color")
                        if traf.swtrails:
                            scr.echo("Trails are currently ON")
                            scr.echo("Trails dt=" + str(traf.trails.dt))
                        else:
                            scr.echo("Trails are currently OFF")
                    else:
                        if cmdargs[1] == "ON":
                            traf.swtrails = True
                            if numargs == 2:
                                try:
                                    trdt = float(cmdargs[2])
                                    traf.trails.dt = trdt
                                except:
                                    scr.echo("TRAIL ON dt")

                        elif cmdargs[1] == "OFF" or cmdargs[1] == "OF":
                            traf.swtrails = False
                            traf.trails.clear()

                        elif len(cmdargs[1]) != 0:
                            correctCommand = True
                            # Check if a color was selected
                            if len(cmdargs) == 2:
                                scr.echo('Syntax error')
                                scr.echo("TRAIL acid color")
                                correctCommand = False

                            if correctCommand:
                                acid = cmdargs[1]
                                color = cmdargs[2]

                                # Does aircraft exist?
                                idx = traf.id2idx(acid)
                                if idx < 0:
                                    scr.echo("TRAILS: " + acid + " not found.")
                                    idx = -1
                                
                                # Does the color exist?
                                if color not in ("BLUE", "RED", "YELLOW"):
                                    scr.echo("Color not found, use BLUE, RED or YELLOW")
                                    idx = -1

                                # Change trail color of aircraft for which there are data
                                if idx >= 0:
                                    traf.changeTrailColor(color, idx)
                                    # scr.echo("TRAIL color of " + acid + " switched to: " + color)
                        else:
                            scr.echo('Syntax error')
                            scr.echo("TRAILS ON/OFF")

                #----------------------------------------------------------------------
                # MCRE n, type/*, alt/*, spd/*, dest/* :Multiple create
                #----------------------------------------------------------------------
                elif cmd[:4] == "MCRE":
                    if numargs == 0:
                        scr.echo("Multiple CREate:")
                        scr.echo("MCRE n, type/*, alt/*, spd/*, dest/*")
                    else:
                        # Currently only n,*,*,*,* supported (or MCRE n)
                        try:
                            n = int(cmdargs[1])

                            if cmdargs[3]!="*":
                                    acalt = txt2alt(cmdargs[3])*ft


                            for i in range(n):
                                acid = "TUD" + str(randint(100, 99999))
                                if actype=="*":
                                    actype = "B744"  # for now
                                else:
                                    actype = cmdargs[2].upper()
                                
                                # Lat/lon.hdg always random on-screen
                                scrlat0,scrlat1,scrlon0,scrlon1 =          \
                                                 scr.getviewlatlon()
                                                 
                                aclat = random() * (scrlat1 - scrlat0) + scrlat0
                                aclon = random() * (scrlon1 - scrlon0) + scrlon0
                                achdg = float(randint(1, 360))

                                # Random altitude
                                if cmdargs[3]=="*":
                                    acalt = float(randint(2000, 39000)) * ft

                                # Speed
                                if cmdargs[4]=="*":
                                    acspd = float(randint(250, 450))
                                else:
                                    acspd = txt2spd(cmdargs[4],h)

                                # Create a/c
                                traf.create(acid, actype, aclat, aclon, achdg, \
                                            acalt, acspd)
                        except:
                            scr.echo('Syntax error')
                            scr.echo("TRAILS ON/OFF")

                #----------------------------------------------------------------------
                # DIST lat1,lon1,lat2,lon2 : 
                #   calculate distance and direction from one pos to 2nd
                #----------------------------------------------------------------------
                elif cmd[:4] == "DIST":
                    if numargs == 0:
                        scr.echo("DIST lat1,lon1,lat2,lon2")
                    else:
                        try:
                            lat0 = float(cmdargs[1])  # lat 1st pos
                            lon0 = float(cmdargs[2])  # lon 1st pos
                            lat1 = float(cmdargs[3])  # lat 2nd pos
                            lon1 = float(cmdargs[4])  # lon 2nd pos
                            qdr, d = qdrdist(lat0, lon0, lat1, lon1)
                            scr.echo("Dist = " + str(round(d,3)) + \
                                 " nm   QDR = " + str(round(qdr,2)) + " deg")
                        except:
                            scr.echo("DIST: Syntax error")
                
                #----------------------------------------------------------------------
                # CALC  expression
                #----------------------------------------------------------------------
                elif cmd[:4] == "CALC":
                    if numargs == 0:
                        scr.echo("CALC expression")
                    else:
                        try:
                            x = eval(cmdline[5:].lower())  # lower for units!
                            scr.echo("Ans = " + str(x))
                        except:
                            scr.echo("CALC: Syntax error")

                #----------------------------------------------------------------------
                # LNAV acid ON/OFF   Switch LNAV (HDG FMS navigation) on/off
                #----------------------------------------------------------------------
                elif cmd == "LNAV":
                    if numargs == 0:
                        scr.echo("LNAV acid, ON/OFF")
                    else:
                        idx = traf.id2idx(cmdargs[1])
                        if cmdargs[2].upper() == "ON":
                            traf.swlnav[idx] = True
                        elif cmdargs[2].upper() == "OFF":
                            traf.swlnav[idx] = False

                #----------------------------------------------------------------------
                # VNAV acid ON/OFF  Switch VNAV (SPD+ALT FMS navigation)  on/off
                #----------------------------------------------------------------------
                elif cmd == "VNAV":
                    if numargs == 0:
                        scr.echo("VNAV acid, ON/OFF")
                    else:
                        idx = traf.id2idx(cmdargs[1])
                        if cmdargs[2].upper() == "ON":
                            traf.swvnav[idx] = True
                        elif cmdargs[2].upper() == "OFF":
                            traf.swvnav[idx] = False


                #----------------------------------------------------------------------
                # ASAS ON/OFF  : switch ASAS on/off
                #----------------------------------------------------------------------
                elif cmd == "ASAS":
                    if numargs == 0:
                        scr.echo("ASAS ON/OFF")
                        if traf.dbconf.swasas:
                            scr.echo("ASAS is currently ON")
                        else:
                            scr.echo("ASAS is currently OFF")
                    else:
                        arg1 = cmdargs[1]  # arguments are strings
                        traf.dbconf.swasas = (arg1.upper() =="ON")
                            
                #----------------------------------------------------------------------
                # ADDWPT   : ADDWPT acid,(WPname / lat,lon),[alt],[spd],[afterwp]
                #----------------------------------------------------------------------
                elif cmd == "ADDWPT":
             
                    if numargs <= 1:
                        scr.echo( \
                        "ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp]")
                    
                    else:
                        acid = cmdargs[1]     # call sign
                        i = traf.id2idx(cmdargs) # Get index of this a/c
                        if traf.id2idx(cmdargs[1])<0:
                            scr.echo("ADDWPT: Aircraft "+cmdargs[1]+" not found.")
                        else:
                            rte = traf.route[i]   # Get pointer to route object

                            # Default values
                            wpok =  False
                            alt = -999
                            spd = -999
                            afterwp = ""
                            if True:

                                # Get waypoint data
                                # Is arg 2 a number? => lat,lon else waypoint name
                                if numargs>3 and cmdargs[2].replace("-","")  \
                                   .replace("+","").replace(".","").isdigit():
      
                                    name    = traf.id[i] # use for wptname
                                    wptype  = rte.wplatlon
                                    lat     = float(cmdargs[2])
                                    lon     = float(cmdargs[3])
                                    if numargs>=4 and cmdargs[4]!="":
                                        alt = txt2alt(cmdargs[4])*ft
                                    if numargs>=5 and cmdargs[5]!="":
                                        spd = txt2spd(cmdargs[5],max(alt,traf.alt[i]))
                                    if numargs>=6:
                                        afterwp = cmdargs[6]
                                    wpok    = True

                                # Is arg navaid/airport/waypoint name?
                                elif numargs>=2:
                                    name    = cmdargs[2]  # search this wpname closest to
                                    wptype  = rte.wpnav
                                    lat     = traf.lat[i]  # a/c position in lat,lon
                                    lon     = traf.lon[i]
                                    if numargs>=3 and cmdargs[3]!="":
                                        alt = txt2alt(cmdargs[3])*ft
                                    if numargs>=4 and cmdargs[4]!="":
                                        spd = txt2spd(cmdargs[4],max(alt,traf.alt[i]))
                                    if numargs>=5:
                                        afterwp = cmdargs[5]
                                    wpok    = True

                                # Add the wpt to route
                                if wpok: 
                                    wpidx = rte.addwpt(traf,i,name,wptype,lat,lon,alt,spd,afterwp)
                                    norig = int(traf.orig[i]!="")
                                    ndest = int(traf.dest[i]!="")

                                    if rte.nwp-norig-ndest==1: # first waypoint: make active
                                       # rte.direct(traf,i,name)
                                       traf.swlnav[i] = True

                                else:
                                    scr.echo(trafid[i]+": waypoint not added")
                                    synerr = True

                #----------------------------------------------------------------------
                # DELWPT   : DELWPT acid,WPname
                #----------------------------------------------------------------------
                elif cmd == "DELWPT":

                    # Help text             
                    if numargs <= 1:
                        scr.echo( \
                        "DELWPT acid, wpname")

                    # Delete waypoint
                    else:
                        acid = cmdargs[1]     # call sign
                        i = traf.id2idx(cmdargs) # Get index of this a/c
                        if traf.id2idx(cmdargs[1])<0:
                            scr.echo("DELWPT: Aircraft "+cmdargs[1]+" not found.")
                        else:
                            rte = traf.route[i]   # Get pointer to route object
                        idx = rte.delwpt(cmdargs[2])
                        if idx==-1:
                            scr.echo("DELWPT: Waypoint "+cmdargs[2]+ \
                                           " not found in route of "+acid+".")

                #----------------------------------------------------------------------
                # DIRECT [TO] wpname: set active waypoint
                #----------------------------------------------------------------------
                elif cmd == "DIRECT":

                    # remove optional 'TO'                   
                    if numargs>=1 and cmdargs[1].upper()=="TO":
                        del cmdargs[1]
                        numargs= numargs-1

                    # Check aircraft id, call route.direct function           
                    if numargs <= 1:
                        scr.echo("DIRECT acid, wpname")
                    else:
                        acid = cmdargs[1]           # call sign
                        i = traf.id2idx(cmdargs[1])    # Get index of this a/c
                        if i<0:
                            scr.echo("DIRECT: Aircraft "+cmdargs[1]+" not found.")

                        else:
                            # Go direct, check success
                           swok,vnavok = traf.route[i].direct(traf,i,cmdargs[2])
                           if swok:
                               traf.swlnav[i] = True
                               traf.swvnav[i] = vnavok
                           else:
                               scr.echo(acid+": DIRECT waypoint"+cmdargs[2] \
                                                 +" not found.")

                #----------------------------------------------------------------------
                #  LISTRTE acid, [pagenr]v   (7 wpts per page)
                #----------------------------------------------------------------------
                elif cmd == "LISTRTE":
                    if numargs < 1:
                        scr.echo("LISTRTE acid [pagenr]")
                    else:
                        acid = cmdargs[1]           # call sign
                        i = traf.id2idx(cmdargs[1])    # Get index of this a/c
                        if i<0:
                            scr.echo("LISTRTE: Aircraft "+cmdargs[1]+" not found.")

                        elif traf.route[i].nwp<=0 :
                            scr.echo("LISTRTE: Aircraft "+cmdargs[1]+" has no route.")
                        else:
                            nwp    = traf.route[i].nwp
                            npages = int((nwp+6)/7)
                            if numargs==1:
                                ipage = 0
                            else:
                                try:
                                    ipage = int(cmdargs[2])
                                except:
                                    synerr = True
                            
                            if not synerr:
                                traf.route[i].listrte(scr,ipage)
                                if ipage+1<npages:
                                    scr.editwin.insert("LISTRTE "+acid+","+str(ipage+1))

                #----------------------------------------------------------------------
                # ECHO: show messages in Edit window
                #----------------------------------------------------------------------
                elif cmd == "ECHO":
                    if numargs == 0:
                        scr.echo("ECHO txt")
                    else:
                        scr.echo(cmdline.strip()[5:])

                #----------------------------------------------------------------------
                # INSEDIT: insert text in Edit window
                #----------------------------------------------------------------------
                elif cmd == "INSEDIT":
                    if numargs == 0:
                        scr.echo("INSEDIT txt")
                    else:
                        scr.editwin.insert(cmdline.strip()[8:])

                #----------------------------------------------------------------------
                # ADSBCOVERAGE: Draw the coverage area of the To70 ADS-B antennas
                #----------------------------------------------------------------------
                elif cmd == "ADSBCOVERAGE":
                    if numargs == 0:
                        scr.echo("ADSBCOVERAGE ON/OFF")
                        if traf.swAdsbCoverage:
                            scr.echo("ADSB Coverage Area is currently ON")
                        else:
                            scr.echo("ADSB Coverage Area is currently OFF")
                    else:
                        if cmdargs[1] == "ON":
                            traf.swAdsbCoverage = True


                        elif cmdargs[1] == "OFF" or cmdargs[1] == "OF":
                            traf.swAdsbCoverage = False

                        else:
                            scr.echo('Syntax error')
                            scr.echo("ADSBCOVERAGE ON/OFF")

                #----------------------------------------------------------------------
                # NOISE on/off: switch Noise on or off
                #----------------------------------------------------------------------
                elif cmd == "NOISE":
                    if numargs == 0:
                        scr.echo("NOISE ON/OFF")
                        if traf.noise:
                            scr.echo("Noise is currently ON")
                        else:
                            scr.echo("Noise is currently OFF")
                    else:
                        if cmdargs[1] == "ON":
                            traf.setNoise(True)

                        elif cmdargs[1] == "OFF" or cmdargs[1] == "OF":
                            traf.setNoise(False)

                        else:
                            scr.echo('Syntax error')
                            scr.echo("NOISE ON/OFF")

                #------------------------------------------------------------------
                # LINE color,lat1,lon1,lat2,lon2 command: draw a line
                #------------------------------------------------------------------
                elif cmd[:4] == "LINE":

                    if numargs == 0:
                        scr.echo("LINE color,lat1,lon1,lat2,lon2")
                        scr.echo("color=red,green,cyan.. (8 basic colours)")
                    else:
                        if numargs==5:
                                icol = col2rgb(cmdargs[1]) # get rgb value for this colour
                                lat1 =  float(cmdargs[2])
                                lon1 =  float(cmdargs[3])
                                lat2 =  float(cmdargs[4])
                                lon2 =  float(cmdargs[5])
                                scr.objappend(1,icol,lat1,lon1,lat2,lon2)


                #------------------------------------------------------------------
                # ENG [acid] Change aircraft's engine
                # available for the internal performance model
                #-------------------------------------------------------------------
                elif cmd == "ENG":
                    if numargs < 1:
                        scr.echo("ENG acid")
                    elif numargs==1:
                        acid = cmdargs[1].upper()
                        idx = traf.id.index(acid)
                        scr.echo("available engine types:")
                        for i in xrange (len(traf.engines[idx])):
                            scr.echo(traf.engines[idx][i])
                            i = i+1
                        scr.echo("Change engine with 'ENG' + [acid] + [id]")
                    elif numargs == 2:
                        acid = cmdargs[1].upper()
                        engid = int(cmdargs[2]) - 1
                        traf.engchange(acid, engid)

                        # self.engchange(acid, engid)

                    else:
                        syntax = False

                #------------------------------------------------------------------
                # DATAFEED CONNECT SERVER_IP_ADDR PORT_NUMBER
                # DATAFEED ON/OFF, DATAFEED DEMO, DATAFEED LOG ON/OFF"
                #------------------------------------------------------------------
                elif cmd[:8] == "DATAFEED":
                    if numargs == 3 and cmdargs[1].upper() == "CONNECT":
                        scr.echo("Connecting to tcp server...")
                        serverip = cmdargs[2]
                        port = int(cmdargs[3]) if cmdargs[3].isdigit() else False
                        if not port:
                            scr.echo("PORT_NUMBER must be valid number")
                        else:
                            self.tmx.df.connect(serverip, port)
                    elif numargs == 1 and cmdargs[1].upper() == "ON":
                        self.tmx.df.start()
                    elif numargs == 1 and cmdargs[1].upper() == "OFF":
                        self.tmx.df.stop()
                    elif numargs == 1 and cmdargs[1].upper() == "DEMO":
                        scr.echo("Use demo server [IP 131.180.117.39, Port 10001]")
                        self.tmx.df.connect('131.180.117.39', 10001)
                    elif numargs == 1 and cmdargs[1].upper() == "LOG":
                        self.tmx.df.set_console_print(True)
                    elif numargs == 2 and cmdargs[1].upper() == "LOG" \
                        and cmdargs[2].upper() == "ON":
                        self.tmx.df.set_console_print(True)
                    elif numargs == 2 and cmdargs[1].upper() == "LOG" \
                        and cmdargs[2].upper() == "OFF":
                        self.tmx.df.set_console_print(False)
                    else:
                        scr.echo("Usage: DATAFEED CONNECT SERVER_IP_ADDR PORT_NUMBER, " +
                            " DATAFEED ON/OFF, DATAFEED DEMO, DATAFEED LOG ON/OFF")

                #------------------------------------------------------------------
                # DUMPRTE acid: Dump the route to the route-file for debugging
                # 
                #------------------------------------------------------------------
                elif cmd[:7] == "DUMPRTE":
                    if numargs == 0:
                        scr.echo("DUMPRTE acid")
                    else:
                        acid = cmdargs[1]
                        i = traf.id2idx(acid)
                        
                        # Open file in append mode, write header
                        f = open("./data/output/routelog.txt","a")
                        f.write("\nRoute "+acid+":\n")
                        f.write("(name,type,lat,lon,alt,spd,toalt,xtoalt)  ")
                        f.write("type: 0=latlon 1=navdb  2=orig  3=dest  4=calwp\n")
                        # write flight plan VNAV data (Lateral is visible on screen)
                        for j in range(traf.route[i].nwp):
                            f.write( str(( j, \
                                  traf.route[i].wpname[j],  \
                                  traf.route[i].wptype[j],  \
                                  round(traf.route[i].wplat[j],4),   \
                                  round(traf.route[i].wplon[j],4),   \
                                  int(0.5+traf.route[i].wpalt[j]/ft),   \
                                  int(0.5+traf.route[i].wpspd[j]/kts),   \
                                  int(0.5+traf.route[i].wptoalt[j]/ft),   \
                                  round(traf.route[i].wpxtoalt[j]/nm,3) \
                                  )) + "\n")

                        # End of data
                        f.write("----\n")
                        f.close()

                #------------------------------------------------------------------
                # !!! This is a template, please make a copy and keep it !!!
                # Insert new command here: first three chars should be unique
                #------------------------------------------------------------------
                elif cmd[:3] == "XXX":
                    if numargs == 0:
                        scr.echo("cmd arg1, arg2")
                    else:
                        arg1 = cmdargs[1]  # arguments are strings
                        arg2 = cmdargs[2]  # arguments are strings

                #-------------------------------------------------------------------
                # Reference to other command files
                # Check external references
                #-------------------------------------------------------------------
                elif cmd[:3] in self.extracmdrefs:
                    self.extracmdrefs[cmd[:3]].process(cmd[3:], numargs, cmdargs, sim, traf, scr, self)

                #-------------------------------------------------------------------
                # Command not found
                #-------------------------------------------------------------------
                else:
                    if numargs==0:
                        scr.echo("Unknown command or aircraft: " + cmd)
                    else:
                        scr.echo("Unknown command: " + cmd)

                #**********************************************************************
                #======================  End of command branches ======================
                #**********************************************************************

                # Syntax not ok, => Default syntax error message:
                if synerr:
                    print "Syntax error in command: ", cmdline
                    scr.echo("Syntax error in command:" + cmd)
                    scr.echo(cmdline)


        # End of for-loop of cmdstack
        self.cmdstack = []
        return