""" Route implementation for the BlueSky FMS."""
from os import path
from numpy import *
import bluesky as bs
from bluesky.tools import geo
from bluesky.core import Replaceable
from bluesky.tools.aero import ft, kts, g0, nm, mach2cas, casormach2tas
from bluesky.tools.misc import degto180, txt2tim, txt2alt, txt2spd
from bluesky.tools.position import txt2pos
from bluesky import stack
from bluesky.stack.cmdparser import Command, command, commandgroup



class Route(Replaceable):
    """
    Route class definition   : Route data for an aircraft
    (basic FMS functionality)

    addwpt(name,wptype,lat,lon,alt) :
    Add waypoint (closest to lat/lon when from navdb

    For lat/lon waypoints: use call sign as wpname, number will be added

    Created by  : Jacco M. Hoekstra
    """

    # Waypoint types:
    wplatlon = 0   # lat/lon waypoint
    wpnav    = 1   # VOR/nav database waypoint
    orig     = 2   # Origin airport
    dest     = 3   # Destination airport
    calcwp   = 4   # Calculated waypoint (T/C, T/D, A/C)
    runway   = 5   # Runway: Copy name and positions

    def __init__(self, acid):
        # Aircraft id (callsign) of the aircraft to which this route belongs
        self.acid = acid
        self.nwp = 0

        # Waypoint data
        self.wpname = []
        self.wptype = []
        self.wplat  = []
        self.wplon  = []
        self.wpalt  = []    # [m] negative value means not specified
        self.wpspd  = []    # [m/s] negative value means not specified
        self.wprta  = []    # [m/s] negative value means not specified
        self.wpflyby = []   # Flyby (True)/flyover(False) switch
        self.wpstack = []   # Stack with command execured when passing this waypoint

        # Made for drones: fly turn mode, means use specified turn radius and optionally turn speed
        self.wpflyturn = []   # Flyturn (True) or flyover/flyby (False) switch
        self.wpturnrad = []   # [nm] Turn radius per waypoint (<0 = not specified)
        self.wpturnspd = []   # [kts] Turn speed (IAS/CAS) per waypoint (<0 = not specified)

        # Current actual waypoint
        self.iactwp = -1

        # Set to default addwpt wpmode
        # Note that neither flyby nor flyturn means: flyover)
        self.swflyby   = True    # Default waypoints are flyby waypoint
        self.swflyturn = False  # Default waypoints are flyby waypoint

        # Default turn values to be used in flyturn mode
        self.turnrad  = -999. # Negative value indicating no value has been set
        self.turnspd  = -999. # Dito, in this case bank angle of vehicle will be used with current speed

        # if the aircraft lands on a runway, the aircraft should keep the
        # runway heading
        # default: False
        self.flag_landed_runway = False

        self.wpdirfrom = []
        self.wpdistto  = []
        self.wpialt    = []
        self.wptoalt   = []
        self.wpxtoalt  = []
        self.wptorta   = []
        self.wpxtorta  = []

    @staticmethod
    def get_available_name(data, name_, len_=2):
        """
        Check if name already exists, if so add integer 01, 02, 03 etc.
        """
        appi = 0  # appended integer to name starts at zero (=nothing)
        # Use Python 3 formatting syntax: "{:03d}".format(7) => "007"
        fmt_ = "{:0" + str(len_) + "d}"

        # Avoid using call sign without number
        if bs.traf.id.count(name_) > 0:
            appi = 1
            name_ = name_+fmt_.format(appi)

        while data.count(name_) > 0 :
            appi += 1
            name_ = name_[:-len_]+fmt_.format(appi)
        return name_

    def addwptStack(self, idx, *args):  # args: all arguments of addwpt
        """ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp],[beforewp]"""

        #debug print ("addwptStack:",args)
        #print("active = ",self.wpname[self.iactwp])
        #print(args)
        # Check FLYBY or FLYOVER switch, instead of adding a waypoint

        if len(args) == 1:
            swwpmode = args[0].replace('-', '')

            if swwpmode == "FLYBY":
                self.swflyby   = True
                self.swflyturn = False
                return True

            elif swwpmode == "FLYOVER":

                self.swflyby   = False
                self.swflyturn = False
                return True

            elif swwpmode == "FLYTURN":

                self.swflyby   = False
                self.swflyturn = True
                return True

        elif len(args) == 2:

            swwpmode = args[0].replace('-', '')

            if swwpmode == "TURNRAD" or swwpmode == "TURNRADIUS":

                try:
                    self.turnrad = float(args[1])/ft # arg was originally parsed as wpalt
                except:
                    return False,"Error in processing value of turn radius"
                return True

            elif swwpmode == "TURNSPD" or swwpmode == "TURNSPEED":

                try:
                    self.turnspd = args[1]*kts/ft # [m/s] Arg was wpalt Keep it as IAS/CAS orig in kts, now in m/s
                except:
                    return False, "Error in processing value of turn speed"

                return True

        # Convert to positions
        name = args[0].upper().strip()

        # Choose reference position ot look up VOR and waypoints
        # First waypoint: own position
        if self.nwp == 0:
            reflat = bs.traf.lat[idx]
            reflon = bs.traf.lon[idx]

        # Or last waypoint before destination
        else:
            if self.wptype[-1] != Route.dest or self.nwp == 1:
                reflat = self.wplat[-1]
                reflon = self.wplon[-1]
            else:
                reflat = self.wplat[-2]
                reflon = self.wplon[-2]

        # Default altitude, speed and afterwp
        alt     = -999.
        spd     = -999.
        afterwp = ""
        beforewp = ""

        # Is it aspecial take-off waypoint?
        takeoffwpt = name.replace('-', '') == "TAKEOFF"

        # Normal waypoint (no take-off waypoint => see else)
        if not takeoffwpt:

            # Get waypoint position
            success, posobj = txt2pos(name, reflat, reflon)
            if success:
                lat      = posobj.lat
                lon      = posobj.lon

                if posobj.type == "nav" or posobj.type == "apt":
                    wptype = Route.wpnav

                elif posobj.type == "rwy":
                    wptype  = Route.runway

                else:  # treat as lat/lon
                    name    = bs.traf.id[idx]
                    wptype  = Route.wplatlon

                if len(args) > 1 and args[1]:
                    alt = args[1]

                if len(args) > 2 and args[2]:
                    spd = args[2]

                if len(args) > 3 and args[3]:
                    afterwp = args[3]

                if len(args) > 4 and args[4]:
                    beforewp = args[4]

            else:
                return False, "Waypoint " + name + " not found."

        # Take off waypoint: positioned 20% of the runway length after the runway
        else:

            # Look up runway in route
            rwyrteidx = -1
            i      = 0
            while i<self.nwp and rwyrteidx<0:
                if self.wpname[i].count("/") >0:
#                   print (self.wpname[i])
                    rwyrteidx = i
                i += 1

            # Only TAKEOFF is specified wihtou a waypoint/runway
            if len(args) == 1 or not args[1]:
                # No runway given: use first in route or current position

                # print ("rwyrteidx =",rwyrteidx)
                # We find a runway in the route, so use it
                if rwyrteidx>0:
                    rwylat   = self.wplat[rwyrteidx]
                    rwylon   = self.wplon[rwyrteidx]
                    aptidx  = bs.navdb.getapinear(rwylat,rwylon)
                    aptname = bs.navdb.aptname[aptidx]

                    rwyname = self.wpname[rwyrteidx].split("/")[1]
                    rwyid = rwyname.replace("RWY","").replace("RW","")
                    rwyhdg = bs.navdb.rwythresholds[aptname][rwyid][2]

                else:
                    rwylat  = bs.traf.lat[idx]
                    rwylon  = bs.traf.lon[idx]
                    rwyhdg = bs.traf.trk[idx]

            elif args[1].count("/") > 0 or len(args) > 2 and args[2]: # we need apt,rwy
                # Take care of both EHAM/RW06 as well as EHAM,RWY18L (so /&, and RW/RWY)
                if args[1].count("/")>0:
                    aptid,rwyname = args[1].split("/")
                else:
                # Runway specified
                    aptid = args[1]
                    rwyname = args[2]

                rwyid = rwyname.replace("RWY", "").replace("RW", "")  # take away RW or RWY
                #                    print ("apt,rwy=",aptid,rwyid)
                # TODO: Add finding the runway heading with rwyrteidx>0 and navdb!!!
                # Try to get it from the database
                try:
                    rwyhdg = bs.navdb.rwythresholds[aptid][rwyid][2]
                except:
                    rwydir = rwyid.replace("L","").replace("R","").replace("C","")
                    try:
                        rwyhdg = float(rwydir)*10.
                    except:
                        return False,name+" not found."

                success, posobj = txt2pos(aptid+"/RW"+rwyid, reflat, reflon)
                if success:
                    rwylat,rwylon = posobj.lat,posobj.lon
                else:
                    rwylat = bs.traf.lat[idx]
                    rwylon = bs.traf.lon[idx]

            else:
                return False,"Use ADDWPT TAKEOFF,AIRPORTID,RWYNAME"

            # Create a waypoint 2 nm away from current point
            rwydist = 2.0 # [nm] use default distance away from threshold
            lat,lon = geo.qdrpos(rwylat, rwylon, rwyhdg, rwydist) #[deg,deg
            wptype  = Route.wplatlon

            # Add after the runwy in the route
            if rwyrteidx > 0:
                afterwp = self.wpname[rwyrteidx]

            elif self.wptype and self.wptype[0] == Route.orig:
                afterwp = self.wpname[0]

            else:
                # Assume we're called before other waypoints are added
                afterwp = ""

            name = "T/O-" + bs.traf.id[idx] # Use lat/lon naming convention
        # Add waypoint
        wpidx = self.addwpt(idx, name, wptype, lat, lon, alt, spd, afterwp, beforewp)

        # Recalculate flight plan
        self.calcfp()

        # Check for success by checking inserted location in flight plan >= 0
        if wpidx < 0:
            return False, "Waypoint " + name + " not added."

        # check for presence of orig/dest
        norig = int(bs.traf.ap.orig[idx] != "") # 1 if orig is present in route
        ndest = int(bs.traf.ap.dest[idx] != "") # 1 if dest is present in route

        # Check whether this is first 'real' waypoint (not orig & dest),
        # And if so, make active
        if self.nwp - norig - ndest == 1:  # first waypoint: make active
            self.direct(idx, self.wpname[norig])  # 0 if no orig
            #print("direct ",self.wpname[norig])
            bs.traf.swlnav[idx] = True

        if afterwp and self.wpname.count(afterwp) == 0:
            return True, "Waypoint " + afterwp + " not found" + \
                "waypoint added at end of route"
        else:
            return True


    def afteraddwptStack(self, idx, *args):  # args: all arguments of addwpt

        # AFTER acid, wpinroute ADDWPT (wpname/lat,lon),[alt],[spd]"
        if len(args) < 3:
            return False, "AFTER needs more arguments"

        # Change order of arguments
        arglst = [args[2], None, None, args[0]]  # postxt,,,afterwp


        # Add alt when given
        if len(args) > 3:
            arglst[1] = args[3]  # alt

        # Add speed when given
        if len(args) > 4:
            arglst[2] = args[4]  # spd

        result = self.addwptStack(idx, *arglst)  # args: all arguments of addwpt

        return result

    def atwptStack(self, idx, *args):  # args: all arguments of addwpt
        #print("args=",args)

        # AT acid, wpinroute [DEL] ALT/SPD spd/alt"
        # args = wpname,SPD/ALT, spd/alt(string)

        if len(args) < 1:
            return False, "AT needs at least an aicraft id and a waypoint name"

        else:
            name = args[0]
            if name in self.wpname:
                wpidx = self.wpname.index(name)

                if len(args) == 1 or \
                        (len(args) == 2 and not args[1].count("/") == 1):
                    # Only show Altitude and/or speed set in route at this waypoint:
                    #    KL204 AT LOPIK => acid AT wpinroute: show alt & spd constraints at this waypoint
                    #    KL204 AT LOPIK SPD => acid AT wpinroute SPD: show spd constraint at this waypoint
                    #    KL204 AT LOPIK ALT => acid AT wpinroute ALT: show alt constraint at this waypoint

                    txt = name + " : "

                    # Select what to show
                    if len(args)==1:
                        swalt = True
                        swspd = True
                        swat  = True
                    else:
                        swalt = args[1].upper()=="ALT"
                        swspd = args[1].upper() in ("SPD","SPEED")
                        swat  = args[1].upper() in ("DO", "STACK")

                        # To be safe show both when we do not know what
                        if not (swalt or swspd or swat):
                            swalt = True
                            swspd = True
                            swat  = True

                    # Show altitude
                    if swalt:
                        if self.wpalt[wpidx] < 0:
                            txt += "-----"

                        elif self.wpalt[wpidx] > 4500 * ft:
                            fl = int(round((self.wpalt[wpidx] / (100. * ft))))
                            txt += "FL" + str(fl)

                        else:
                            txt += str(int(round(self.wpalt[wpidx] / ft)))

                        if swspd:
                            txt += "/"

                    # Show speed
                    if swspd:
                        if self.wpspd[wpidx] < 0:
                            txt += "---"
                        else:
                            txt += str(int(round(self.wpspd[wpidx] / kts)))

                    # Type
                    if swalt and swspd:
                        if self.wptype[wpidx] == Route.orig:
                            txt += "[orig]"
                        elif self.wptype[wpidx] == Route.dest:
                            txt += "[dest]"

                    # Show also stacked commands for when passing this waypoint
                    if swat:
                        if len(self.wpstack[wpidx])>0:
                            txt = txt+"\nStack:\n"
                            for stackedtxt in self.wpstack[wpidx]:
                                txt = txt + stackedtxt + "\n"


                    return True, txt

                elif args[1].count("/")==1:
                    # Set both alt & speed at this waypoint
                    #     KL204 AT LOPIK FL090/250  => acid AT wpinroute alt/spd
                    success = True

                    # Use parse from stack.py to interpret alt & speed
                    alttxt, spdtxt = args[1].split('/')

                    # Edit waypoint altitude constraint
                    if alttxt.count('-') > 1: # "----" = delete
                        self.wpalt[wpidx]  = -999.
                    else:
                        try:
                            self.wpalt[wpidx] = txt2alt(alttxt)
                        except ValueError as e:
                            success = False

                    # Edit waypoint speed constraint
                    if spdtxt.count('-') > 1: # "----" = delete
                        self.wpspd[wpidx]  = -999.
                    else:
                        try:
                            self.wpalt[wpidx] = txt2spd(spdtxt)
                        except ValueError as e:
                            success = False

                    if not success:
                        return False,"Could not parse "+args[1]+" as alt / spd"

                    # If success: update flight plan and guidance
                    self.calcfp()
                    self.direct(idx, self.wpname[self.iactwp])

                #acid AT wpinroute ALT/SPD alt/spd
                elif len(args)>=3:
                    # KL204 AT LOPIK ALT FL090 => set altitude to be reached at this waypoint in route
                    # KL204 AT LOPIK SPD 250 => Set speed at twhich is set at this waypoint
                    # KL204 AT LOPIK DO PAN LOPIK => When passing stack command after DO
                    # KL204 AT LOPIK STACK PAN LOPIK => AT...STACK synonym for AT...DO
                    # KL204 AT LOPIK DO ALT FL240 => => stack "KL204 ALT FL240" => use acid from beginning if omitted as first argument

                    swalt = args[1].upper()=="ALT"
                    swspd = args[1].upper() in ("SPD","SPEED")
                    swat  = args[1].upper() in ("DO","STACK")

                    # Use parse from stack.py to interpret alt & speed

                    # Edit waypoint altitude constraint
                    if swalt:
                        try:
                            self.wpalt[wpidx] = txt2alt(args[2])
                        except ValueError as e:
                            return False, e.args[0]

                    # Edit waypoint speed constraint
                    elif swspd:
                        try:
                            self.wpspd[wpidx] = txt2spd(args[2])
                        except ValueError as e:
                            return False, e.args[0]

                    # add stack command: args[1] is DO or STACK, args[2:] contains a command
                    elif swat:
                        # Check if first argument is missing aircraft id, if so, use this acid

                        # IF command starts with aircraft id, it is not missing
                        cmd = args[2].upper()
                        if not(cmd in bs.traf.id):
                            # Look up arg types
                            try:
                                cmdobj = Command.cmddict.get(cmd)

                                # Command found, check arguments
                                argtypes = cmdobj.annotations

                                if argtypes[0]=="acid" and not (args[3].upper() in bs.traf.id):
                                    # missing acid, so add ownship acid
                                    self.wpstack[wpidx].append(self.acid+" "+" ".join(args[2:]))
                                else:
                                    # This command does not need an acid or it is already first argument
                                    self.wpstack[wpidx].append(" ".join(args[2:]))
                            except:
                                return False, "Stacked command "+cmd+"unknown"
                        else:
                            # Command line starts with an aircraft id at the beginning of the command line, stack it
                            self.wpstack[wpidx].append(" ".join(args[2:]))

                    # Delete a constraint (or both) at this waypoint
                    elif args[1]=="DEL" or args[1]=="DELETE" or args[1]=="CLR" or args[1]=="CLEAR" :
                        swalt = args[2].upper()=="ALT"
                        swspd = args[2].upper() in ("SPD","SPEED")
                        swboth  = args[2].upper()=="BOTH"
                        swall   = args[2].upper()=="ALL"

                        if swspd or swboth or swall:
                            self.wpspd[wpidx]  = -999.

                        if swalt or swboth or swall:
                            self.wpalt[wpidx]  = -999.

                        if swall:
                            self.wpstack[wpidx]=[]

                    else:
                        return False,"No "+args[1]+" at ",name


                    # If success: update flight plan and guidance
                    self.calcfp()
                    self.direct(idx, self.wpname[self.iactwp])

            # Waypoint not found in route
            else:
                return False, name + " not found in route " + bs.traf.id[idx]

        return True

    def overwrite_wpt_data(self, wpidx, wpname, wplat, wplon, wptype, wpalt,
                           wpspd):
        """
        Overwrites information for a waypoint, via addwpt_data/9
        """

        self.addwpt_data(True, wpidx, wpname, wplat, wplon, wptype, wpalt,
                         wpspd)

    def insert_wpt_data(self, wpidx, wpname, wplat, wplon, wptype, wpalt,
                        wpspd):
        """
        Inserts information for a waypoint, via addwpt_data/9
        """

        self.addwpt_data(False, wpidx, wpname, wplat, wplon, wptype, wpalt,
                         wpspd)

    def addwpt_data(self, overwrt, wpidx, wpname, wplat, wplon, wptype,
                    wpalt, wpspd):
        """
        Overwrites or inserts information for a waypoint
        """
        wplat = (wplat + 90.) % 180. - 90.
        wplon = (wplon + 180.) % 360. - 180.

        if overwrt:
            self.wpname[wpidx]  = wpname
            self.wplat[wpidx]   = wplat
            self.wplon[wpidx]   = wplon
            self.wpalt[wpidx]   = wpalt
            self.wpspd[wpidx]   = wpspd
            self.wptype[wpidx]  = wptype
            self.wpflyby[wpidx] = self.swflyby
            self.wpflyturn[wpidx] = self.swflyturn
            self.wpturnrad[wpidx] = self.turnrad
            self.wpturnspd[wpidx] = self.turnspd
            self.wprta[wpidx]   = -999.0 # initially no RTA
            self.wpstack[wpidx] = []

        else:
            self.wpname.insert(wpidx, wpname)
            self.wplat.insert(wpidx, wplat)
            self.wplon.insert(wpidx, wplon)
            self.wpalt.insert(wpidx, wpalt)
            self.wpspd.insert(wpidx, wpspd)
            self.wptype.insert(wpidx, wptype)
            self.wpflyby.insert(wpidx, self.swflyby)
            self.wpflyturn.insert(wpidx, self.swflyturn)
            self.wpturnrad.insert(wpidx, self.turnrad)
            self.wpturnspd.insert(wpidx, self.turnspd)
            self.wprta.insert(wpidx,-999.0)       # initially no RTA
            self.wpstack.insert(wpidx,[])


    def addwpt(self, iac, name, wptype, lat, lon, alt=-999., spd=-999., afterwp="", beforewp=""):
        """Adds waypoint an returns index of waypoint, lat/lon [deg], alt[m]"""
#        print ("addwpt:")
#        print ("iac = ",iac)
#        print ("name = "+name)
#        print ("alt = ",alt)
#        print ("spd = ",spd)
#        print ("afterwp ="+afterwp)
#        print
        # For safety
        self.nwp = len(self.wplat)

        name = name.upper().strip()

        wplat = lat
        wplon = lon

        # Be default we trust, distrust needs to be earned
        wpok = True   # switch for waypoint check

        # Check if name already exists, if so add integer 01, 02, 03 etc.
        wprtename = Route.get_available_name(
            self.wpname, name)
        # Select on wptype
        # ORIGIN: Wptype is origin/destination?
        if wptype == Route.orig or wptype == Route.dest:
            orig = wptype == Route.orig
            wpidx = 0 if orig else -1
            suffix = "ORIG" if orig else "DEST"

            if not name == bs.traf.id[iac] + suffix:  # published identifier
                i = bs.navdb.getaptidx(name)
                if i >= 0:
                    wplat = bs.navdb.aptlat[i]
                    wplon = bs.navdb.aptlon[i]

            if not orig and alt < 0:
                alt = 0

            # Overwrite existing origin/dest
            if self.nwp > 0 and self.wptype[wpidx] == wptype:
                self.overwrite_wpt_data(
                    wpidx, wprtename, wplat, wplon, wptype, alt, spd)

            # Or add before first waypoint/append to end
            else:
                if not orig:
                    wpidx = len(self.wplat)

                self.insert_wpt_data(
                    wpidx, wprtename, wplat, wplon, wptype, alt, spd)

                self.nwp += 1
                if orig and self.iactwp >= 0:
                    self.iactwp += 1
                elif not orig and self.iactwp < 0 and self.nwp == 1:
                    # When only waypoint: adjust pointer to point to destination
                    self.iactwp = 0

            idx = 0 if orig else self.nwp - 1

        # NORMAL: Wptype is normal waypoint? (lat/lon or nav)
        else:
            # Lat/lon: wpname is then call sign of aircraft: add number
            if wptype == Route.wplatlon:
                newname = Route.get_available_name(
                    self.wpname, name, 3)

            # Else make data complete with nav database and closest to given lat,lon
            else: # so wptypewpnav
                newname = wprtename

                if not wptype == Route.runway:
                    i = bs.navdb.getwpidx(name, lat, lon)
                    wpok = (i >= 0)

                    if wpok:
                        wplat = bs.navdb.wplat[i]
                        wplon = bs.navdb.wplon[i]
                    else:
                        i = bs.navdb.getaptidx(name)
                        wpok = (i >= 0)
                        if wpok:
                            wplat = bs.navdb.aptlat[i]
                            wplon = bs.navdb.aptlon[i]

            # Check if afterwp or beforewp is specified and found:
            aftwp = afterwp.upper().strip()  # Remove space, upper case
            bfwp = beforewp.upper().strip()

            if wpok:

                if (afterwp and self.wpname.count(aftwp) > 0) or \
                        (beforewp and self.wpname.count(bfwp) > 0):

                    wpidx = self.wpname.index(aftwp) + 1 if afterwp else \
                        self.wpname.index(bfwp)

                    self.insert_wpt_data(
                        wpidx, newname, wplat, wplon, wptype, alt, spd)

                    if afterwp and self.iactwp >= wpidx:
                        self.iactwp += 1

                # No afterwp: append, just before dest if there is a dest
                else:
                    # Is there a destination?
                    if self.nwp > 0 and self.wptype[-1] == Route.dest:
                        wpidx = self.nwp - 1
                    else:
                        wpidx = self.nwp

                    self.addwpt_data(
                        False, wpidx, newname, wplat, wplon, wptype, alt, spd)

                idx = wpidx
                self.nwp += 1

            else:
                idx = -1
                if len(self.wplat) == 1:
                    self.iactwp = 0

            #update qdr in traffic
            bs.traf.actwp.next_qdr[iac] = self.getnextqdr()

        # Update waypoints
        if not (wptype == Route.calcwp):
            self.calcfp()

        # Update autopilot settings
        if wpok and 0 <= self.iactwp < self.nwp:
            self.direct(iac, self.wpname[self.iactwp])


        return idx

    def beforeaddwptStack(self, idx, *args):  # args: all arguments of addwpt
        # BEFORE acid, wpinroute ADDWPT acid, (wpname/lat,lon),[alt],[spd]"
        if len(args) < 3:
            return False, "BEFORE needs more arguments"

        # Change order of arguments
        arglst = [args[2], None, None, None, args[0]]  # postxt,,,,beforewp

        # Add alt when given
        if len(args) > 3:
            arglst[1] = args[3]  # alt

        # Add speed when given
        if len(args) > 4:
            arglst[2] = args[4]  # spd

        result = self.addwptStack(idx, *arglst)  # args: all arguments of addwpt

        return result

    def direct(self, idx, wpnam):
        #print("Hello from direct")
        """Set active point to a waypoint by name"""
        name = wpnam.upper().strip()
        if name != "" and self.wpname.count(name) > 0:
            wpidx = self.wpname.index(name)

            self.iactwp = wpidx

            bs.traf.actwp.lat[idx]    = self.wplat[wpidx]
            bs.traf.actwp.lon[idx]    = self.wplon[wpidx]
            bs.traf.actwp.flyby[idx]  = self.wpflyby[wpidx]
            bs.traf.actwp.flyturn[idx] = self.wpflyturn[wpidx]
            bs.traf.actwp.turnrad[idx] = self.wpturnrad[wpidx]
            bs.traf.actwp.turnspd[idx] = self.wpturnspd[wpidx]

            # Do calculation for VNAV
            self.calcfp()

            bs.traf.actwp.xtoalt[idx] = self.wpxtoalt[wpidx]
            bs.traf.actwp.nextaltco[idx] = self.wptoalt[wpidx]

            bs.traf.actwp.torta[idx]    = self.wptorta[wpidx]    # available for active RTA-guidance
            bs.traf.actwp.xtorta[idx]  = self.wpxtorta[wpidx]  # available for active RTA-guidance

            #VNAV calculations like V/S and speed for RTA
            bs.traf.ap.ComputeVNAV(idx, self.wptoalt[wpidx], self.wpxtoalt[wpidx],\
                                        self.wptorta[wpidx],self.wpxtorta[wpidx])

            # If there is a speed specified, process it
            if self.wpspd[wpidx]>0.:
                # Set target speed for autopilot

                if self.wpalt[wpidx] < 0.0:
                    alt = bs.traf.alt[idx]
                else:
                    alt = self.wpalt[wpidx]

                # Check for valid Mach or CAS
                if self.wpspd[wpidx] <2.0:
                    cas = mach2cas(self.wpspd[wpidx], alt)
                else:
                    cas = self.wpspd[wpidx]

                # Save it for next leg
                bs.traf.actwp.nextspd[idx] = cas

            # No speed specified for next leg
            else:
                bs.traf.actwp.nextspd[idx] = -999.


            qdr, dist = geo.qdrdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])

            if self.wpflyturn[wpidx] or self.wpturnrad[wpidx]<0.:
                turnrad = self.wpturnrad[wpidx]
            else:
                turnrad = bs.traf.tas[idx]*bs.traf.tas[idx]/tan(radians(25.)) / g0 / nm  # [nm]default bank angle 25 deg


            bs.traf.actwp.turndist[idx] = (bs.traf.actwp.flyby[idx] > 0.5)  *   \
                     turnrad*abs(tan(0.5*radians(max(5., abs(degto180(qdr -
                        self.wpdirfrom[self.iactwp]))))))    # [nm]


            bs.traf.swlnav[idx] = True
            return True
        else:
            return False, "Waypoint " + wpnam + " not found"


    def SetRTA(self, idx, name, txt):  # all arguments of setRTA
        """SetRTA acid, wpname, time: add RTA to waypoint record"""
        timeinsec = txt2tim(txt)
        #print(timeinsec)
        if name in self.wpname:
            wpidx = self.wpname.index(name)
            self.wprta[wpidx] = timeinsec
            #print("Ik heb",self.wprta[wpidx],"op",self.wpname[wpidx],"gezet!")

            # Recompute route and update actwp because of RTA addition
            self.direct(idx, self.wpname[self.iactwp])

        return True

    def listrte(self, idx, ipage=0):
        """LISTRTE command: output route to screen"""
        if self.nwp <= 0:
            return False, "Aircraft has no route."

        if idx<0:
            return False, "Aircraft id not found."

        for i in range(ipage * 7, ipage * 7 + 7):
            if 0 <= i < self.nwp:
                # Name
                if i == self.iactwp:
                    txt = "*" + self.wpname[i] + " : "
                else:
                    txt = " " + self.wpname[i] + " : "

                # Altitude
                if self.wpalt[i] < 0:
                    txt += "-----/"

                elif self.wpalt[i] > 4500 * ft:
                    fl = int(round((self.wpalt[i] / (100. * ft))))
                    txt += "FL" + str(fl) + "/"

                else:
                    txt += str(int(round(self.wpalt[i] / ft))) + "/"

                # Speed
                if self.wpspd[i] < 0.:
                    txt += "---"
                elif self.wpspd[i] > 2.0:
                    txt += str(int(round(self.wpspd[i] / kts)))
                else:
                    txt += "M" + str(self.wpspd[i])

                # Type: orig, dest, C = flyby, | = flyover, U = flyturn
                if self.wptype[i] == Route.orig:
                    txt += "[orig]"
                elif self.wptype[i] == Route.dest:
                    txt += "[dest]"
                elif self.wpflyturn[i]:
                    txt += "[U]"
                elif self.wpflyby[i]:
                    txt += "[C]"
                else: # FLYOVER
                    txt += "[|]"


                # Display message
                bs.scr.echo(txt)

        # Add command for next page to screen command line
        npages = int((self.nwp + 6) / 7)
        if ipage + 1 < npages:
            bs.scr.cmdline("LISTRTE " + bs.traf.id[idx] + "," + str(ipage + 1))

    def getnextwp(self):
        """Go to next waypoint and return data"""

        if self.flag_landed_runway:

            # when landing, LNAV is switched off
            lnavon = False

            # no further waypoint
            nextqdr = -999.

            # and the aircraft just needs a fixed heading to
            # remain on the runway
            # syntax: HDG acid,hdg (deg,True)
            name = self.wpname[self.iactwp]

            # Change RW06,RWY18C,RWY24001 to resp. 06,18C,24
            if "RWY" in name:
                rwykey = name[8:10]
                if not name[10].isdigit():
                    rwykey = rwykey+name[10]
            # also if it is only RW
            else:
                rwykey = name[7:9]
                if not name[9].isdigit():
                    rwykey = rwykey+name[9]

            wphdg = bs.navdb.rwythresholds[name[:4]][rwykey][2]

            # keep constant runway heading
            stack.stack("HDG " + str(self.acid) + " " + str(wphdg))

            # start decelerating
            stack.stack("DELAY " + "10 " + "SPD " + str(self.acid) + " " + "10")

            # delete aircraft
            stack.stack("DELAY " + "42 " + "DEL " + str(self.acid))

            return self.wplat[self.iactwp],self.wplon[self.iactwp],   \
                           self.wpalt[self.iactwp],self.wpspd[self.iactwp],   \
                           self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp], \
                           self.wpxtorta[self.iactwp], self.wptorta[self.iactwp], \
                           lnavon,self.wpflyby[self.iactwp], \
                           self.wpflyturn[self.iactwp],self.wpturnrad[self.iactwp],\
                           self.wpturnspd[self.iactwp], \
                           nextqdr

        lnavon = self.iactwp +1 < self.nwp
        if lnavon:
            self.iactwp += 1

        nextqdr = self.getnextqdr()

        # in case that there is a runway, the aircraft should remain on it
        # instead of deviating to the airport centre
        # When there is a destination: current = runway, next  = Dest
        # Else: current = runway and this is also the last waypoint
        if (self.wptype[self.iactwp] == 5 and
                self.wpname[self.iactwp] == self.wpname[-1]) or \
           (self.wptype[self.iactwp] == 5 and self.iactwp+1<self.nwp and
                self.wptype[self.iactwp + 1] == 3):

            self.flag_landed_runway = True

        #print ("getnextwp:",self.wpname[self.iactwp],"   torta = ",self.wptorta[self.iactwp])

        return self.wplat[self.iactwp],self.wplon[self.iactwp],   \
               self.wpalt[self.iactwp],self.wpspd[self.iactwp],   \
               self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp],\
               self.wpxtorta[self.iactwp],self.wptorta[self.iactwp],\
               lnavon,self.wpflyby[self.iactwp], \
               self.wpflyturn[self.iactwp], self.wpturnrad[self.iactwp], \
               self.wpturnspd[self.iactwp], \
               nextqdr

    def runactwpstack(self):
        for cmdline in self.wpstack[self.iactwp]:
            stack.stack(cmdline)
            #debug
            # stack.stack("ECHO "+self.acid+" AT "+self.wpname[self.iactwp]+" command issued:"+cmdline)
        return

    def delrte(self,iac=None):
        """Delete complete route"""
        # Simple re-initialize this route as empty
        self.__init__(bs.traf.id[iac])

        # Also disable LNAV,VNAV if route is deleted
        if self.nwp == 0 and (iac or iac == 0):
            bs.traf.swlnav[iac]    = False
            bs.traf.swvnav[iac]    = False
            bs.traf.swvnavspd[iac] = False

        return True

    def delwpt(self,delwpname,iac=None):
        """Delete waypoint"""

        # Delete complete route?
        if delwpname =="*":
            return self.delrte(iac)

        # Look up waypoint
        idx = -1
        i = len(self.wpname)
        while idx == -1 and i > 0:
            i -= 1
            if self.wpname[i].upper() == delwpname.upper():
                idx = i

        # check if active way point is the one being deleted and that it is not the last wpt.
        # If active wpt is deleted then change path of aircraft
        if self.iactwp == idx and not idx == self.nwp - 1:
            self.direct(iac, self.wpname[idx + 1])

        # Delete waypoint
        if idx == -1:
            return False, "Waypoint " + delwpname + " not found"

        self.nwp =self.nwp - 1
        del self.wpname[idx]
        del self.wplat[idx]
        del self.wplon[idx]
        del self.wpalt[idx]
        del self.wpspd[idx]
        del self.wprta[idx]
        del self.wptype[idx]
        if self.iactwp > idx:
            self.iactwp = max(0, self.iactwp - 1)

        self.iactwp = min(self.iactwp, self.nwp - 1)

        # If no waypoints left, make sure to disable LNAV/VNAV
        if self.nwp==0 and (iac or iac==0):
            bs.traf.swlnav[iac]    =  False
            bs.traf.swvnav[iac]    =  False
            bs.traf.swvnavspd[iac] =  False

        return True

    def newcalcfp(self): # Not used for now: alternative way which use T/C and T/D as waypoints
        """Do flight plan calculations"""

        # Remove old top of descents and old top of climbs
        while self.wpname.count("T/D")>0:
            self.delwpt("T/D")

        while self.wpname.count("T/C")>0:
            self.delwpt("T/C")

        # Remove old actual position waypoints
        while self.wpname.count("A/C")>0:
            self.delwpt("A/C")

        # Insert actual position as A/C waypoint
        acidx = bs.traf.id2idx(self.acid)
        idx = self.iactwp
        self.insertcalcwp(idx,"A/C")
        self.wplat[idx] = bs.traf.lat[acidx] # deg
        self.wplon[idx] = bs.traf.lon[acidx] # deg
        self.wpalt[idx] = bs.traf.alt[acidx] # m
        self.wpspd[idx] = bs.traf.tas[acidx] # m/s

        # Calculate distance to last waypoint in route
        nwp = len(self.wpname)
        dist2go = [0.0]
        for i in range(nwp - 2, -1, -1):
            qdr, dist = geo.qdrdist(self.wplat[i], self.wplon[i],
                                    self.wplat[i + 1], self.wplon[i + 1])
            dist2go = [dist2go[0] + dist] + dist2go

        # Make VNAV WP list with only waypoints with altitude constraints
        # This list we will use to find where to insert t/c and t/d
        alt = []
        x   = []
        name = []
        for i in range(nwp):
            if self.wpalt[i]>-1.:
                alt.append(self.wpalt[i])
                x.append(dist2go[i])
                name.append(self.wpname[i]+" ")    # space for check first 2 chars later

        # Find where to insert cruise segment (if any)

        # Find longest segment without altitude constraints

        desslope = clbslope = 1.0
        crzalt = bs.traf.crzalt[acidx]
        if crzalt>0.:
            ilong  = -1
            dxlong = 0.0

            nvwp = len(alt)
            for i in range(nvwp-1):
                if x[i]-x[i+1]> dxlong:
                    ilong  = i
                    dxlong = x[i]-x[i+1]

            # VNAV parameters to insert T/Cs and T/Ds
            crzdist  = 20.*nm   # minimally required distance at cruise level
            clbslope = 3000.*ft/(10.*nm)    # 1:3 rule for now
            desslope = clbslope             # 1:3 rule for now

            # Can we get a sufficient distance at cruise altitude?
            if max(alt[ilong],alt[ilong+1]) < crzalt :
                dxclimb = (crzalt-alt[ilong])*clbslope
                dxdesc  = (crzalt-alt[ilong+1])*desslope
                if x[ilong] - x[ilong+1] > dxclimb + crzdist + dxdesc:

                    # Insert T/C (top of climb) at cruise level
                   name.insert(ilong+1,"T/C")
                   alt.insert(ilong+1,crzalt)
                   x.insert(ilong+1,x[ilong]+dxclimb)

                    # Insert T/D (top of descent) at cruise level
                   name.insert(ilong+2,"T/D")
                   alt.insert(ilong+2,crzalt)
                   x.insert(ilong+2,x[ilong+1]-dxdesc)

        # Compare angles to rates:
        epsh = 50.*ft   # Nothing to be done for small altitude changes
        epsx = 1.*nm    # [m] Nothing to be done at this short range
        i = 0
        while i<len(alt)-1:
            if name[i][:2]=="T/":
                continue

            dy = alt[i+1]-alt[i]   # alt change (pos = climb)
            dx = x[i]-x[i+1]       # distance (positive)

            dxdes = abs(dy)/desslope
            dxclb = abs(dy)/clbslope

            if dy<epsh and  dx + epsx > dxdes:   # insert T/D?

               name.insert(i+1,"T/D")
               alt.insert(i+1,alt[i])
               x.insert(i+1,x[i+1]-dxdes)
               i += 1

            elif dy>epsh and  dx + epsx > dxclb:  # insert T/C?

               name.insert(i+1,"T/C")
               alt.insert(i+1,alt[i+1])
               x.insert(i+1,x[i]+dxclb)
               i += 2
            else:
                i += 1

        # Now insert T/Cs and T/Ds in actual flight plan
        nvwp = len(alt)
        for i in range(nvwp,-1,-1):

            # Copy all new waypoints (which are all named T/C or T/D)
            if name[i][:2]=="T/":

                # Find place in flight plan to insert T/C or T/D
                j = nvwp-1
                while dist2go[j]<x[i] and j>1:
                    j=j-1

                # Interpolation factor for position on leg
                f   = (x[i]-dist2go[j+1])/(dist2go[j]-dist2go[j+1])

                lat = f*self.wplat[j]+(1.-f)*self.wplat[j+1]
                lon = f*self.wplon[j]+(1.-f)*self.wplon[j+1]

                self.wpname.insert(j,name[i])
                self.wptype.insert(j,Route.calcwp)
                self.wplat.insert(j,lat)
                self.wplon.insert(j,lon)
                self.wpalt.insert(j,alt[i])
                self.wpspd.insert(j,-999.)

    def insertcalcwp(self, i, name):
        """Insert empty wp with no attributes at location i"""

        self.wpname.insert(i,name)
        self.wplat.insert(i,0.)
        self.wplon.insert(i,0.)
        self.wpalt.insert(i,-999.)
        self.wpspd.insert(i,-999.)
        self.wptype.insert(i,Route.calcwp)

    def calcfp(self): # Current Flight Plan calculations, which actualize based on flight condition
        """Do flight plan calculations"""
#        self.delwpt("T/D")
#        self.delwpt("T/C")

        # Direction to waypoint
        self.nwp = len(self.wpname)

        # Create flight plan calculation table
        self.wpdirfrom   = self.nwp*[0.]
        self.wpdistto    = self.nwp*[0.]
        self.wpialt      = self.nwp*[-1]
        self.wptoalt     = self.nwp*[-999.]
        self.wpxtoalt    = self.nwp*[1.]  # Avoid division by zero
        self.wpirta      = self.nwp*[-1]
        self.wptorta     = self.nwp*[-999.]
        self.wpxtorta    = self.nwp*[1.]  #[m] Avoid division by zero

        # No waypoints: make empty variables to be safe and return: nothing to do
        if self.nwp==0:
            return

        # Calculate lateral leg data
        # LNAV: Calculate leg distances and directions

        for i in range(0, self.nwp - 1):
            qdr,dist = geo.qdrdist(self.wplat[i]  ,self.wplon[i],
                                self.wplat[i+1],self.wplon[i+1])
            self.wpdirfrom[i] = qdr
            self.wpdistto[i+1]  = dist #[nm]  distto is in nautical miles

        if self.nwp>1:
            self.wpdirfrom[-1] = self.wpdirfrom[-2]

        # Calculate longitudinal leg data
        # VNAV: calc next altitude constraint: index, altitude and distance to it
        ialt = -1     # index to waypoint with next altitude constraint
        toalt = -999. # value of next altitude constraint
        xtoalt = 0.   # distance to next altitude constraint from this wp
        for i in range(self.nwp-1,-1,-1):

            # waypoint with altitude constraint (dest of al specified)
            if self.wptype[i]==Route.dest:
                ialt   = i
                toalt  = 0.
                xtoalt = 0.                # [m]

            elif self.wpalt[i] >= 0:
                ialt   = i
                toalt  = self.wpalt[i]
                xtoalt = 0.                # [m]

            # waypoint with no altitude constraint:keep counting
            else:
                if i!=self.nwp-1:
                    xtoalt = xtoalt + self.wpdistto[i+1]*nm  # [m] xtoalt is in meters!
                else:
                    xtoalt = 0.0

            self.wpialt[i]   = ialt
            self.wptoalt[i]  = toalt   #[m]
            self.wpxtoalt[i] = xtoalt  #[m]

        # RTA: calc next rta constraint: index, altitude and distance to it
        # If any RTA.
        if any(array(self.wprta)>=0.0):
            #print("Yes, I found RTAs")
            irta = -1       # index of wp
            torta = -999.   # next rta value
            xtorta = 0.     # distance to next rta
            for i in range(self.nwp - 1, -1, -1):

                # waypoint with rta: reset counter, update rts
                if self.wprta[i] >= 0:
                    irta = i
                    torta = self.wprta[i]
                    xtorta = 0.  # [m]

                # waypoint with no altitude constraint:keep counting
                else:
                    if i != self.nwp - 1:
                        # No speed or rta constraint: add to xtorta
                        if self.wpspd[i] <= 0.0:
                            xtorta = xtorta + self.wpdistto[i + 1] * nm  # [m] xtoalt is in meters!
                        else:
                            # speed constraint on this leg: shift torta to account for this
                            # altitude unknown
                            if self.wptoalt[i] >0.:
                                alt = toalt
                            else:
                                # TODO: current a/c altitude would be better guess, but not accessible here
                                # as we do not know aircraft index for this route
                                alt = 10000.*ft # default to minimize errors, when no alt constraints are present
                            legtas = casormach2tas(self.wpspd[i],alt)
                            #TODO: account for wind at this position vy adding wind vectors to waypoints?

                            # xtorta stays the same! This leg will not be available for RTA scheduling, so distance
                            # is not in xtorta. Therefore we need to subtract legtime to ignore this leg for the RTA
                            # scheduling
                            legtime = self.wpdistto[i+1]/legtas
                            torta = torta - legtime
                    else:
                        xtorta = 0.0
                        torta = -999.0

                self.wpirta[i]   = irta
                self.wptorta[i]  = torta  # [s]
                self.wpxtorta[i] = xtorta  # [m]
            #print("wpxtorta=",self.wpxtorta)
            #print("wptorta=", self.wptorta)

    def findact(self,i):
        """ Find best default active waypoint.
        This function is called during route creation"""
        #print "findact is called.!"

        # Check for easy answers first
        if self.nwp<=0:
            return -1

        elif self.nwp == 1:
            return 0

        # Find closest
        wplat  = array(self.wplat)
        wplon  = array(self.wplon)
        dy = (wplat - bs.traf.lat[i])
        dx = (wplon - bs.traf.lon[i]) * bs.traf.coslat[i]
        dist2 = dx*dx + dy*dy
        # Note: the max() prevents walking back, even in cases when this might be apropriate,
        # such as when previous waypoints have been deleted

        iwpnear = max(self.iactwp,argmin(dist2))

        #Unless behind us, next waypoint?
        if iwpnear+1<self.nwp:
            qdr = degrees(arctan2(dx[iwpnear],dy[iwpnear]))
            delhdg = abs(degto180(bs.traf.trk[i]-qdr))

            # we only turn to the first waypoint if we can reach the required
            # heading before reaching the waypoint
            time_turn = max(0.01,bs.traf.tas[i])*radians(delhdg)/(g0*tan(bs.traf.bank[i]))
            time_straight= sqrt(dist2[iwpnear])*60.*nm/max(0.01,bs.traf.tas[i])

            if time_turn > time_straight:
                iwpnear += 1

        return iwpnear

    def dumpRoute(self, idx):
        acid = bs.traf.id[idx]
        # Open file in append mode, write header
        with open(path.join(bs.settings.log_path, 'routelog.txt'), "a") as f:
            f.write("\nRoute "+acid+":\n")
            f.write("(name,type,lat,lon,alt,spd,toalt,xtoalt)  ")
            f.write("type: 0=latlon 1=navdb  2=orig  3=dest  4=calwp\n")

            # write flight plan VNAV data (Lateral is visible on screen)
            for j in range(self.nwp):
                f.write( str(( j, self.wpname[j], self.wptype[j],
                      round(self.wplat[j], 4), round(self.wplon[j], 4),
                      int(0.5+self.wpalt[j]/ft), int(0.5+self.wpspd[j]/kts),
                      int(0.5+self.wptoalt[j]/ft), round(self.wpxtoalt[j]/nm, 3)
                      )) + "\n")

            # End of data
            f.write("----\n")
            f.close()

    def getnextqdr(self):
        # get qdr for next leg
        if -1 < self.iactwp < self.nwp - 1:
            nextqdr, dist = geo.qdrdist(\
                        self.wplat[self.iactwp],  self.wplon[self.iactwp],\
                        self.wplat[self.iactwp+1],self.wplon[self.iactwp+1])
        else:
            nextqdr = -999.
        return nextqdr
