""" Route implementation for the BlueSky FMS."""
from os import path
from numpy import *
import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.aero import ft, kts, g0, nm, mach2cas
from bluesky.tools.misc import degto180
from bluesky.tools.position import txt2pos
from bluesky import stack
from bluesky.stack import Argparser

# Register settings defaults
bs.settings.set_variable_defaults(log_path='output')

class Route:
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

    def __init__(self):
        self.nwp    = 0

        # Waypoint data
        self.wpname = []
        self.wptype = []
        self.wplat  = []
        self.wplon  = []
        self.wpalt  = []    # [m] negative value means not specified
        self.wpspd  = []    # [m/s] negative value means not specified
        self.wpflyby = []   # Flyby (True)/flyover(False) switch

        # Current actual waypoint
        self.iactwp = -1
        self.swflyby  = True  # Default waypoints are flyby waypoint

        # if the aircraft lands on a runway, the aircraft should keep the
        # runway heading
        # default: False
        self.flag_landed_runway = False

        self.iac = self.wpdirfrom = self.wpdistto = self.wpialt = \
            self.wptoalt = self.wpxtoalt = None

    @staticmethod
    def get_available_name(data, name_, len_=2):
        """
        Check if name already exists, if so add integer 01, 02, 03 etc.
        """
        appi = 0  # appended integer to name starts at zero (=nothing)
        nameorg = name_
        while data.count(name_) > 0:
            appi += 1
            format_ = "%s%0" + str(len_) + "d"
            name_ = format_ % (nameorg, appi)
        return name_

    def addwptStack(self, idx, *args):  # args: all arguments of addwpt
        """ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp],[beforewp]"""

#        print "addwptStack:",args
        # Check FLYBY or FLYOVER switch, instead of adding a waypoint

        if len(args) == 1:
            isflyby = args[0].replace('-', '')

            if isflyby == "FLYBY":
                self.swflyby = True
                return True

            elif isflyby == "FLYOVER":
                self.swflyby = False
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
#                    print (self.wpname[i])
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
                # TDO: Add fingind the runway heading with rwyrteidx>0 and navdb!!!
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

        # Check for success by checking insetred locaiton in flight plan >= 0
        if wpidx < 0:
            return False, "Waypoint " + name + " not added."

        # chekc for presence of orig/dest
        norig = int(bs.traf.ap.orig[idx] != "")
        ndest = int(bs.traf.ap.dest[idx] != "")

        # Check whether this is first 'real' wayppint (not orig & dest),
        # And if so, make active
        if self.nwp - norig - ndest == 1:  # first waypoint: make active
            self.direct(idx, self.wpname[norig])  # 0 if no orig
            bs.traf.swlnav[idx] = True

        if afterwp and self.wpname.count(afterwp) == 0:
            return True, "Waypoint " + afterwp + " not found" + \
                "waypoint added at end of route"
        else:
            return True


    def afteraddwptStack(self, idx, *args):  # args: all arguments of addwpt

        # AFTER acid, wpinroute ADDWPT acid, (wpname/lat,lon),[alt],[spd]"
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

        # AT acid, wpinroute [DEL] ALT/SPD spd/alt"
        # args = wpname,SPD/ALT, spd/alt(string)

        if len(args) < 1:
            return False, "AT needs at least an aicraft id and a waypoint name"

        else:
            name = args[0]
            if name in self.wpname:
                wpidx = self.wpname.index(name)

                # acid AT wpinroute: show alt & spd constraints at this waypoint
                # acid AT wpinroute SPD: show spd constraint at this waypoint
                # acid AT wpinroute ALT: show alt constraint at this waypoint

                if len(args) == 1 or \
                        (len(args) == 2 and not args[1].count("/") == 1):

                    txt = name + " : "

                    # Select what to show
                    if len(args)==1:
                        swalt = True
                        swspd = True
                    else:
                        swalt = args[1].upper()=="ALT"
                        swspd = args[1].upper() in ("SPD","SPEED")

                        # To be safe show both when we do not know what
                        if not (swalt or swspd):
                            swalt = True
                            swspd = True

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

                    return True, txt

                elif args[1].count("/")==1:
                    # acid AT wpinroute alt"/"spd
                    success = True

                    # Use parse from stack.py to interpret alt & speed
                    alttxt, spdtxt = args[1].split('/')

                    # Edit waypoint altitude constraint
                    if alttxt.count('-') > 1: # "----" = delete
                        self.wpalt[wpidx]  = -999.
                    else:
                        parser = Argparser(['alt'], [False], alttxt)
                        if parser.parse():
                            self.wpalt[wpidx] = parser.arglist[0]
                        else:
                            success = False

                    # Edit waypoint speed constraint
                    if spdtxt.count('-') > 1: # "----" = delete
                        self.wpspd[wpidx]  = -999.
                    else:
                        parser = Argparser(['spd'], [False], spdtxt)
                        if parser.parse():
                            self.wpspd[wpidx] = parser.arglist[0]
                        else:
                            success = False

                    if not success:
                        return False,"Could not parse "+args[1]+" as alt / spd"

                    # If success: update flight plan and guidance
                    self.calcfp()
                    self.direct(idx, self.wpname[self.iactwp])


                #acid AT wpinroute ALT/SPD alt/spd
                elif len(args)==3 :
                    swalt = args[1].upper()=="ALT"
                    swspd = args[1].upper() in ("SPD","SPEED")

                    # Use parse from stack.py to interpret alt & speed

                    # Edit waypoint altitude constraint
                    if swalt:
                        parser = Argparser(['alt'], [False], args[2])
                        if parser.parse():
                            self.wpalt[wpidx] = parser.arglist[0]
                        else:
                            return False,'Could not parse "' + args[2] + '" as altitude'

                    # Edit waypoint speed constraint
                    elif swspd:
                        parser = Argparser(['spd'], [False], args[2])
                        if parser.parse():
                            self.wpspd[wpidx] = parser.arglist[0]
                        else:
                            return False,'Could not parse "' + args[2] + '" as speed'

                    # Delete a constraint (or both) at this waypoint
                    elif args[1]=="DEL" or args[1]=="DELETE":
                        swalt = args[2].upper()=="ALT"
                        swspd = args[2].upper() in ("SPD","SPEED")
                        both  = args[2].upper() in ("ALL","BOTH")

                        if swspd or both:
                            self.wpspd[wpidx]  = -999.

                        if swalt or both:
                            self.wpalt[wpidx]  = -999.

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
                           wpspd, swflyby):
        """
        Overwrites information for a waypoint, via addwpt_data/9
        """

        self.addwpt_data(True, wpidx, wpname, wplat, wplon, wptype, wpalt,
                         wpspd, swflyby)

    def insert_wpt_data(self, wpidx, wpname, wplat, wplon, wptype, wpalt,
                        wpspd, swflyby):
        """
        Inserts information for a waypoint, via addwpt_data/9
        """

        self.addwpt_data(False, wpidx, wpname, wplat, wplon, wptype, wpalt,
                         wpspd, swflyby)

    def addwpt_data(self, overwrt, wpidx, wpname, wplat, wplon, wptype,
                    wpalt, wpspd, swflyby):
        """
        Overwrites or inserts information for a waypoint
        """
        wplat = (wplat + 90.) % 180. - 90.
        wplon = (wplon + 180.) % 360. - 180.

        if overwrt:
            self.wpname[wpidx] = wpname
            self.wplat[wpidx] = wplat
            self.wplon[wpidx] = wplon
            self.wpalt[wpidx] = wpalt
            self.wpspd[wpidx] = wpspd
            self.wptype[wpidx] = wptype
            self.wpflyby[wpidx] = swflyby
        else:
            self.wpname.insert(wpidx, wpname)
            self.wplat.insert(wpidx, wplat)
            self.wplon.insert(wpidx, wplon)
            self.wpalt.insert(wpidx, wpalt)
            self.wpspd.insert(wpidx, wpspd)
            self.wptype.insert(wpidx, wptype)
            self.wpflyby.insert(wpidx, swflyby)


    def addwpt(self, iac, name, wptype, lat, lon, alt=-999., spd=-999., afterwp="", beforewp=""):
        """Adds waypoint an returns index of waypoint, lat/lon [deg], alt[m]"""
#        print ("addwpt:")
#        print ("iac = ",iac)
#        print ("name = "+name)
#        print ("alt = ",alt)
#        print ("spd = ",spd)
#        print ("afterwp ="+afterwp)
#        print
        self.iac = iac    # a/c to which this route belongs
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
                    wpidx, wprtename, wplat, wplon, wptype, alt, spd,
                    self.swflyby)

            # Or add before first waypoint/append to end
            else:
                if not orig:
                    wpidx = len(self.wplat)

                self.insert_wpt_data(
                    wpidx, wprtename, wplat, wplon, wptype, alt, spd,
                    self.swflyby)

                self.nwp += 1
                if orig and self.iactwp > 0:
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
                        wpidx, newname, wplat, wplon, wptype, alt, spd,
                        self.swflyby)

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
                        False, wpidx, newname, wplat, wplon, wptype, alt, spd,
                        self.swflyby)

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
        """Set active point to a waypoint by name"""
        name = wpnam.upper().strip()
        if name != "" and self.wpname.count(name) > 0:
            wpidx = self.wpname.index(name)
            self.iactwp = wpidx

            bs.traf.actwp.lat[idx]   = self.wplat[wpidx]
            bs.traf.actwp.lon[idx]   = self.wplon[wpidx]
            bs.traf.actwp.flyby[idx] = self.wpflyby[wpidx]


            self.calcfp()
            bs.traf.ap.ComputeVNAV(idx, self.wptoalt[wpidx], self.wpxtoalt[wpidx])

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
                bs.traf.actwp.spd[idx] = cas

                # When already in VNAV: fly it
                if bs.traf.swvnav[idx]:
                    bs.traf.selspd[idx]=cas

            # No speed specified for next leg
            else:
                 bs.traf.actwp.spd[idx] = -999.


            qdr, dist = geo.qdrdist(bs.traf.lat[idx], bs.traf.lon[idx],
                                bs.traf.actwp.lat[idx], bs.traf.actwp.lon[idx])

            turnrad = bs.traf.tas[idx]*bs.traf.tas[idx]/tan(radians(25.)) / g0 / nm  # [nm]default bank angle 25 deg

            bs.traf.actwp.turndist[idx] = (bs.traf.actwp.flyby[idx] > 0.5)  *   \
                     turnrad*abs(tan(0.5*radians(max(5., abs(degto180(qdr -
                        self.wpdirfrom[self.iactwp]))))))    # [nm]


            bs.traf.swlnav[idx] = True
            return True
        else:
            return False, "Waypoint " + wpnam + " not found"

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

                # Type
                if self.wptype[i] == Route.orig:
                    txt += "[orig]"
                elif self.wptype[i] == Route.dest:
                    txt += "[dest]"

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
                if not name[10].isidigit():
                    rwykey = rwykey+name[10]
            # also if it is only RW
            else:
                rwykey = name[7:9]
                if not name[9].isidigit():
                    rwykey = rwykey+name[9]

            wphdg = bs.navdb.rwythresholds[name[:4]][rwykey][2]

            # keep constant runway heading
            stack.stack("HDG " + str(bs.traf.id[self.iac]) + " " + str(wphdg))

            # start decelerating
            stack.stack("DELAY " + "10 " + "SPD " + str(bs.traf.id[self.iac]) + " " + "10")

            # delete aircraft
            stack.stack("DELAY " + "42 " + "DEL " + str(bs.traf.id[self.iac]))

            return self.wplat[self.iactwp],self.wplon[self.iactwp],   \
                           self.wpalt[self.iactwp],self.wpspd[self.iactwp],   \
                           self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp],\
                           lnavon,self.wpflyby[self.iactwp], nextqdr

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

#        print ("getnextwp:",self.wpname[self.iactwp])

        return self.wplat[self.iactwp],self.wplon[self.iactwp],   \
               self.wpalt[self.iactwp],self.wpspd[self.iactwp],   \
               self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp],\
               lnavon,self.wpflyby[self.iactwp], nextqdr

    def delrte(self,iac=None):
        """Delete complete route"""
        # Simple re-initialize this route as empty
        self.__init__()

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

        # Delete waypoint
        if idx == -1:
            return False, "Waypoint " + delwpname + " not found"

        self.nwp =self.nwp - 1
        del self.wpname[idx]
        del self.wplat[idx]
        del self.wplon[idx]
        del self.wpalt[idx]
        del self.wpspd[idx]
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

    def newcalcfp(self):
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
        idx = self.iactwp
        self.insertcalcwp(idx,"A/C")
        self.wplat[idx] = bs.traf.lat[self.iac] # deg
        self.wplon[idx] = bs.traf.lon[self.iac] # deg
        self.wpalt[idx] = bs.traf.alt[self.iac] # m
        self.wpspd[idx] = bs.traf.tas[self.iac] # m/s

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

        desslope = clbslope = 1.
        crzalt = bs.traf.crzalt[self.iac]
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

    def calcfp(self):
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
        self.wpxtoalt    = self.nwp*[1.]

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

        # Calclate longitudinal leg data
        # VNAV: calc next altitude constraint: index, altitude and distance to it
        ialt = -1
        toalt = -999.
        xtoalt = 0.
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
                    xtoalt += self.wpdistto[i+1]*nm  # [m] xtoalt is in meters!
                else:
                    xtoalt = 0.0

            self.wpialt[i]   = ialt
            self.wptoalt[i]  = toalt   #[m]
            self.wpxtoalt[i] = xtoalt  #[m]

    def findact(self,i):
        """ Find best default active waypoint.
        This function is called during route creation"""
#        print "findact is called.!"

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
