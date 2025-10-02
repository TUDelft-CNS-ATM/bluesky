""" Route implementation for the BlueSky FMS."""
import math
from weakref import WeakValueDictionary
import numpy as np
import bluesky as bs
from bluesky.tools import geo
from bluesky.core import Base
from bluesky.tools.aero import ft, kts, g0, nm, mach2cas, casormach2tas
from bluesky.tools.misc import degto180, txt2tim, txt2alt, txt2spd, cas2tas
from bluesky.tools.position import txt2pos
from bluesky import stack
from bluesky.stack.cmdparser import Command, command, commandgroup



class Route(Base):
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

    # Aircraft route objects
    _routes: WeakValueDictionary[str, 'Route'] = WeakValueDictionary()

    def __init__(self, acid):
        super().__init__()
        # Add self to dictionary of all aircraft routes
        Route._routes[acid] = self
        # Aircraft id (callsign) of the aircraft to which this route belongs
        self.acid = acid
        self.nwp = 0

       # Waypoint data
        self.wpname = []    # List of waypoint names for this flight plan
        self.wptype = []    # List of waypoint types
        self.wplat  = []    # List of waypoint latitudes
        self.wplon  = []    # List of waypoint longitudes
        self.wpalt  = []    # [m] negative value means not specified
        self.wpspd  = []    # [m/s] negative value means not specified
        self.wprta  = []    # [m/s] negative value means not specified
        self.wpflyby = []   # Flyby (True)/flyover(False) switch
        self.wpstack = []   # Stack with command execured when passing this waypoint

        # Made for drones: fly turn mode, means use specified turn radius and optionally turn speed
        self.wpflyturn  = []   # Flyturn (True) or flyover/flyby (False) switch
        self.wpturnbank = []   # [deg] Bank angle per waypoint
        self.wpturnrad  = []   # [nm] Turn radius per waypoint (<0 = not specified)
        self.wpturnspd  = []   # [kts] Turn speed (IAS/CAS) per waypoint (<0 = not specified)
        self.wpturnhdgr = [] # [deg/s] Heading rate, uses actual speed to calculate bank & radius (<0 = not specified)

        # Current actual waypoint
        self.iactwp = -1

        # Set to default addwpt wpmode
        # Note that neither flyby nor flyturn means: flyover)
        self.swflyby   = True    # Default waypoints are flyby waypoint
        self.swflyturn = False  # Default waypoints are waypoints w/o specified turn

        # Default turn values to be used in flyturn mode
        self.bank = 25.
        self.turnbank  = -999.   # [deg] Negative value indicating no value has been set
        self.turnrad   = -999. # [m] Negative value indicating no value has been set
        self.turnspd   = -999. # [kts] Dito, in this case bank angle of vehicle will be used with current speed
        self.turnhdgr  = -999. # [deg/s] Dito, in this case bank angle of vehicle will be used with current speed
        # The last two defined turn properties. The list will contain variable names, with the latest
        # Changed being at index 1, and the oldest changed being at index 0.
        # For example, I changed SPD and RAD in that order. The list will look like this:
        # acrte_2_defined = ['acrte.turnspd', 'acrte.turnrad']
        # I then change turnbank. The list will now look like this:
        # acrte_2_defined = ['acrte.turnrad', 'acrte.turnbank']
        self.last_2_defined = []

        # if the aircraft lands on a runway, the aircraft should keep the
        # runway heading
        # default: False
        self.flag_landed_runway = False

        self.wpdirfrom = []  # [deg] direction leg to wp
        self.wpdirto   = []  # [deg] direction leg from wp
        self.wpdistto  = []  # [nm] leg length to wp
        self.wpialt    = []
        self.wptoalt   = []  # [m] next alt contraint
        self.wpxtoalt  = []  # [m] distance ot next alt constraint
        self.wptorta   = []  # [s] next time constraint
        self.wpxtorta  = []  # [m] distance to next time constaint

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
    
    @stack.command(name = 'ADDWPTMODE')
    @staticmethod
    def addwptMode(acidx: 'acid', mode: 'txt' = None, value: float = None):
        ''' Changes the mode of the ADDWPT command to add waypoints of type 'mode'.
            The mode specified will be used for all ADDWPT commands that are called
            after this command.
            
            Arguments:
            - acid: Aircraft id
            - mode: The mode to use for future waypoints. Options are:
                    FLYBY/FLYOVER/FLYTURN/TURNSPD/TURNRAD/TURNHDG
        '''
        # Get aircraft route
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        # First, we want to check what 'mode' is, and then call addwptStack 
        # accordingly.
        if mode in ['FLYBY', 'FLYOVER', 'FLYTURN']:
            # We're just changing addwpt mode, call the appropriate function.
            Route.addwptStack(acidx, mode)
            return True
        
        elif mode in  ['TURNSPEED', 'TURNSPD', 'TURNRADIUS', 'TURNRAD', 'TURNBANK', 'TURNPHI', \
                       'TURNHDGRATE', 'TURNHDG','TURNHDGR']:
            # We're changing the turn properties.
            Route.addwptStack(acidx, mode, value)
            return True
            
        elif mode == None:
            # Just echo the current wptmode
            if acrte.swflyby == True and acrte.swflyturn == False:
                stack.echo('Current ADDWPT mode is FLYBY.')
                return True

            elif acrte.swflyby == False and acrte.swflyturn == False:
                stack.echo('Current ADDWPT mode is FLYOVER.')
                return True

            else:
                stack.echo('Current ADDWPT mode is FLYTURN.')
                return True
            
    @stack.command(name='ADDWPT', annotations='acid,wpt,[alt,spd,wpinroute,wpinroute]', aliases=("WPTYPE",))
    @staticmethod
    def addwptStack(acidx, *args):  # args: all arguments of addwpt
        """ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp],[beforewp]"""
        # First get the appropriate ac route
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        
        #debug print ("addwptStack:",args)
        #print("active = ",self.wpname[self.iactwp])
        #print(argsnwp
        # Check FLYBY or FLYOVER switch, instead of adding a waypoint

        if len(args) == 1:
            swwpmode = args[0].replace('-', '')

            if swwpmode == "FLYBY":
                acrte.swflyby   = True
                acrte.swflyturn = False
                return True

            elif swwpmode == "FLYOVER":
                acrte.swflyby   = False
                acrte.swflyturn = False
                return True

            elif swwpmode == "FLYTURN":
                acrte.swflyby   = False
                acrte.swflyturn = True
                return True

        elif len(args) == 2:

            swwpmode = args[0].replace('-', '')

            if swwpmode == "TURNRAD" or swwpmode == "TURNRADIUS":
                try:
                    if args[1]=="OFF":
                        acrte.turnrad = -999
                        # If this is in last 2 defined, pop it
                        if 'acrte.turnrad' in acrte.last_2_defined:
                            acrte.last_2_defined.pop(acrte.last_2_defined.index('acrte.turnrad'))
                    else:
                        acrte.turnrad = float(args[1]/ft*nm) #arg was originally parsed as wpalt
                        # Well, last we defined is turnrad, so append it to the list
                        acrte.last_2_defined.append('turnrad')
                        # Now, unless the first value is also acrte.turnrad, we set that var to -999 again
                        if len(acrte.last_2_defined) > 2:
                            # Reset first variable if one of the remaining 3
                            if acrte.last_2_defined[0] == 'turnhdgr':
                                acrte.turnhdgr = -999.
                            elif acrte.last_2_defined[0] == 'turnbank':
                                acrte.turnbank = -999.
                            elif acrte.last_2_defined[0] == 'turnspd':
                                acrte.turnspd = -999.
                            # Pop the first one
                            acrte.last_2_defined.pop(0)
                        # Lastly, there is a possibility that we are left with a list of duplicates. Fix it
                        if len(acrte.last_2_defined) == 2 and acrte.last_2_defined[0] == acrte.last_2_defined[1]:
                            # Duplicates, pop the first one
                            acrte.last_2_defined.pop(0)
                        
                except:
                    return False,"Error in processing value of turn radius"

                # Switch flyturn automatically when this is set
                acrte.swflyby = False
                acrte.swflyturn = True

                return True

            elif swwpmode == "TURNSPD" or swwpmode == "TURNSPEED":

                try:
                    if args[1] == "OFF":
                        acrte.turnspd = -999
                    else:
                        acrte.turnspd = args[1]*kts/ft # [m/s] Arg was wpalt Keep it as IAS/CAS orig in kts, now in m/s
                        # Well, last we defined is turnspd, so append it to the list
                        acrte.last_2_defined.append('turnspd')
                        # Now, unless the first value is also acrte.turnspd, we set that var to -999 again
                        if len(acrte.last_2_defined) > 2:
                            # Reset first variable if one of the remaining 3
                            if acrte.last_2_defined[0] == 'turnhdgr':
                                acrte.turnhdgr = -999.
                            elif acrte.last_2_defined[0] == 'turnbank':
                                acrte.turnbank = -999.
                            elif acrte.last_2_defined[0] == 'turnrad':
                                acrte.turnrad = -999.
                            # Pop the first one
                            acrte.last_2_defined.pop(0)
                        # Lastly, there is a possibility that we are left with a list of duplicates. Fix it
                        if len(acrte.last_2_defined) == 2 and acrte.last_2_defined[0] == acrte.last_2_defined[1]:
                            # Duplicates, pop the first one
                            acrte.last_2_defined.pop(0)
                            
                except:
                    return False, "Error in processing value of turn speed"

                # Switch flyturn automatically when this is set
                acrte.swflyby = False
                acrte.swflyturn = True
                
                return True


            elif swwpmode == "TURNHDGRATE" or swwpmode == "TURNHDG" or swwpmode == "TURNHDGR":
                
                try:
                    if args[1] == "OFF":
                        acrte.turnhdgr = -999
                    else:
                        acrte.turnhdgr = args[1]/ft # [deg/s] turn rate
                        # Well, last we defined is turnhdgr, so append it to the list
                        acrte.last_2_defined.append('turnhdgr')
                        # Now, unless the first value is also acrte.turnrad, we set that var to -999 again
                        if len(acrte.last_2_defined) > 2:
                            # Reset first variable if one of the remaining 3
                            if acrte.last_2_defined[0] == 'turnbank':
                                acrte.turnbank = -999.
                            elif acrte.last_2_defined[0] == 'turnspd':
                                acrte.turnspd = -999.
                            elif acrte.last_2_defined[0] == 'turnrad':
                                acrte.turnrad = -999.
                            # Pop the first one
                            acrte.last_2_defined.pop(0)
                        # Lastly, there is a possibility that we are left with a list of duplicates. Fix it
                        if len(acrte.last_2_defined) == 2 and acrte.last_2_defined[0] == acrte.last_2_defined[1]:
                            # Duplicates, pop the first one
                            acrte.last_2_defined.pop(0)
                            
                except:
                    return False, "Error in processing value of turn heading rate"

                # Switch flyturn automatically when this is set
                acrte.swflyby = False
                acrte.swflyturn = True

                return True
            
            elif swwpmode == 'TURNBANK' or swwpmode == 'TURNPHI':

                try:
                    if args[1] == "OFF":
                        acrte.turnbank = -999
                    else:                    
                        # Cap the bank angle between 0 and 90 degrees
                        bankangle = max(0, min(args[1]/ft, 90))
                        acrte.turnbank = bankangle # [deg] desired bank angle
                        
                        # Well, last we defined is turnbank, so append it to the list
                        acrte.last_2_defined.append('turnbank')
                        # Now, unless the first value is also acrte.turnbank, we set that var to -999 again
                        if len(acrte.last_2_defined) > 2:
                            # Reset first variable if one of the remaining 3
                            if acrte.last_2_defined[0] == 'turnhdgr':
                                acrte.turnhdgr = -999.
                            elif acrte.last_2_defined[0] == 'turnspd':
                                acrte.turnspd = -999.
                            elif acrte.last_2_defined[0] == 'turnrad':
                                acrte.turnrad = -999.
                            # Pop the first one
                            acrte.last_2_defined.pop(0)
                        # Lastly, there is a possibility that we are left with a list of duplicates. Fix it
                        if len(acrte.last_2_defined) == 2 and acrte.last_2_defined[0] == acrte.last_2_defined[1]:
                            # Duplicates, pop the first one
                            acrte.last_2_defined.pop(0)
                            
                except:
                    return False, "Error in processing value of turn heading rate"

                # Switch flyturn automatically when this is set
                acrte.swflyby = False
                acrte.swflyturn = True
                
                return True
                    


        # Convert to positions
        name = args[0].upper().strip()

        # Choose reference position ot look up VOR and waypoints
        # First waypoint: own position
        if acrte.nwp == 0:
            reflat = bs.traf.lat[acidx]
            reflon = bs.traf.lon[acidx]

        # Or last waypoint before destination
        else:
            if acrte.wptype[-1] != Route.dest or acrte.nwp == 1:
                reflat = acrte.wplat[-1]
                reflon = acrte.wplon[-1]
            else:
                reflat = acrte.wplat[-2]
                reflon = acrte.wplon[-2]

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
                    name    = acid
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
            while i<acrte.nwp and rwyrteidx<0:
                if acrte.wpname[i].count("/") >0:
                    rwyrteidx = i
                i += 1

            # Only TAKEOFF is specified wihtou a waypoint/runway
            if len(args) == 1 or not args[1]:
                # No runway given: use first in route or current position

                # print ("rwyrteidx =",rwyrteidx)
                # We find a runway in the route, so use it
                if rwyrteidx>0:
                    rwylat   = acrte.wplat[rwyrteidx]
                    rwylon   = acrte.wplon[rwyrteidx]
                    aptidx  = bs.navdb.getapinear(rwylat,rwylon)
                    aptname = bs.navdb.aptname[aptidx]

                    rwyname = acrte.wpname[rwyrteidx].split("/")[1]
                    rwyid = rwyname.replace("RWY","").replace("RW","")
                    rwyhdg = bs.navdb.rwythresholds[aptname][rwyid][2]

                else:
                    rwylat  = bs.traf.lat[acidx]
                    rwylon  = bs.traf.lon[acidx]
                    rwyhdg = bs.traf.trk[acidx]

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
                    rwylat = bs.traf.lat[acidx]
                    rwylon = bs.traf.lon[acidx]

            else:
                return False,"Use ADDWPT TAKEOFF,AIRPORTID,RWYNAME"

            # Create a waypoint 2 nm away from current point
            rwydist = 2.0 # [nm] use default distance away from threshold
            lat,lon = geo.qdrpos(rwylat, rwylon, rwyhdg, rwydist) #[deg,deg
            wptype  = Route.wplatlon

            # Add after the runwy in the route
            if rwyrteidx > 0:
                afterwp = acrte.wpname[rwyrteidx]

            elif acrte.wptype and acrte.wptype[0] == Route.orig:
                afterwp = acrte.wpname[0]

            else:
                # Assume we're called before other waypoints are added
                afterwp = ""

            name = "T/O-" + acid # Use lat/lon naming convention
        # Add waypoint
        wpidx = acrte.addwpt(acidx, name, wptype, lat, lon, alt, spd, afterwp, beforewp)

        # Recalculate flight plan
        acrte.calcfp()

        # Check for success by checking inserted location in flight plan >= 0
        if wpidx < 0:
            return False, "Waypoint " + name + " not added."

        # check for presence of orig/dest
        norig = int(bs.traf.ap.orig[acidx] != "") # 1 if orig is present in route
        ndest = int(bs.traf.ap.dest[acidx] != "") # 1 if dest is present in route

        # Check whether this is first 'real' waypoint (not orig & dest),
        # And if so, make active
        if acrte.nwp - norig - ndest == 1:  # first waypoint: make active
            acrte.direct(acidx, acrte.wpname[norig])  # 0 if no orig
            #print("direct ",self.wpname[norig])
            bs.traf.swlnav[acidx] = True

        if afterwp and acrte.wpname.count(afterwp) == 0:
            print(afterwp, acrte.wpname)
            return True, "Waypoint " + afterwp + " not found\n" + \
                "waypoint added at end of route"
        else:
            return True

    @stack.command
    def addwaypoints(acidx: 'acid', *args):
        ''' ADDWAYPOINTS: Add multiple waypoints to the route of an aircraft.
        
            Arguments:
            - acid: Aircraft id
            - lat_i: Latitude of the ith waypoint
            - lon_i: Longitude of the ith waypoint
            - alt_i: Altitude constraint at the ith waypoint
            - spd_i: Speed constraint at the ith waypoint
            - type_i: Turn type of the ith waypoint. [TURNSPD/TURNRAD/FLYBY]
            - turnspd_i/turnrad_i: Turn speed / turn radius constraint at the ith waypoint
            '''
        # Args come in this order: lat, lon, alt, spd, TURNSPD/TURNRAD/FLYBY, turnspeed or turnrad value
        # If turn is '0', then ignore turnspeed
        if len(args)%6 !=0:
            stack.echo('You missed a waypoint value, arguement number must be a multiple of 6.')
            return

        # Get and reset current aircraft route
        acid = bs.traf.id[acidx]
        acrte = Route._routes.get(acid)

        args = np.reshape(args, (int(len(args)/5), 5))

        for wpdata in args:
            # Get needed values
            lat = float(wpdata[0]) # deg
            lon = float(wpdata[1]) # deg
            if wpdata[2]:
                alt = txt2alt(wpdata[2]) # comes in feet, convert
            else:
                alt = -999
            if wpdata[3]:
                spd = txt2spd(wpdata[3])
            else:
                spd = -999

            # Do flyby or flyturn or flyover processing
            if wpdata[4] == 'FLYTURN': #
                acrte.swflyby   = False
                acrte.swflyturn = True

            elif wpdata[4] == 'FLYBY':
                acrte.swflyby   = True
                acrte.swflyturn = False

            elif wpdata[4] == 'FLYOVER':
                acrte.swflyby   = False
                acrte.swflyturn = False

            else:
                # Either it's a flyby, or a typo.
                acrte.swflyby   = True
                acrte.swflyturn = False

            wpidx = acrte.addwpt_simple(acidx, acid, Route.wplatlon, lat, lon, alt, spd)
        
        # Direct to first waypoint
        acrte.direct(acidx, acrte.wpname[0])  # 0 if no orig
        #print("direct ",self.wpname[norig])
        bs.traf.swlnav[acidx] = True

        # Calculate flight plan
        acrte.calcfp()

        # Check for success by checking inserted location in flight plan >= 0
        if wpidx < 0:
            return False, "Waypoint " + acid + " not added."
        
        # # Direct aircraft to first waypoint
        # acrte.iactwp = 0
        # acrte.direct(acidx, acrte.wpname[0])
        

    def addwpt_simple(self, iac, name, wptype, lat, lon, alt=-999., spd=-999.):
        """Adds waypoint in the most simple way possible"""
        # For safety
        self.nwp = len(self.wplat)

        name = name.upper().strip()

        wplat = lat
        wplon = lon

        # Check if name already exists, if so add integer 01, 02, 03 etc.
        newname = Route.get_available_name(
            self.wpname, name, 3)

        self.addwpt_data(
            False, self.nwp, newname, wplat, wplon, wptype, alt, spd)

        idx = self.nwp
        self.nwp += 1

        #update qdr and "last waypoint switch" in traffic
        if idx>=0:
            bs.traf.actwp.next_qdr[iac] = self.getnextqdr()
            bs.traf.actwp.swlastwp[iac] = (self.iactwp==self.nwp-1)
            
        # Update autopilot settings
        if 0 <= self.iactwp < self.nwp:
            self.direct(iac, self.wpname[self.iactwp])

        return idx

    @stack.command
    @staticmethod
    def before(acidx : 'acid', beforewp: 'wpinroute', addwpt, waypoint, alt: 'alt' = None, spd: 'spd' = None):
        ''' BEFORE acid, wpinroute ADDWPT acid, (wpname/lat,lon),[alt],[spd]

            Before waypoint, add a waypoint to route of aircraft (FMS).
        '''
        return Route.addwptStack(acidx, waypoint, alt, spd, None, beforewp)

    @stack.command
    @staticmethod
    def after(acidx: 'acid', afterwp: 'wpinroute', addwpt, waypoint, alt:'alt' = None, spd: 'spd' = None):
        ''' AFTER acid, wpinroute ADDWPT (wpname/lat,lon),[alt],[spd]

            After waypoint, add a waypoint to route of aircraft (FMS).
        '''
        return Route.addwptStack(acidx, waypoint, alt, spd, afterwp)

    @stack.command
    @staticmethod
    def at(acidx: 'acid', atwp : 'wpinroute', *args):
        ''' AT acid, wpinroute [DEL] ALT/SPD/DO alt/spd/stack command'''
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        if atwp in acrte.wpname:
            wpidx = acrte.wpname.index(atwp)

            if not args or \
                    (len(args) == 1 and not args[0].count("/") == 1):
                # Only show Altitude and/or speed set in route at this waypoint:
                #    KL204 AT LOPIK => acid AT wpinroute: show alt & spd constraints at this waypoint
                #    KL204 AT LOPIK SPD => acid AT wpinroute SPD: show spd constraint at this waypoint
                #    KL204 AT LOPIK ALT => acid AT wpinroute ALT: show alt constraint at this waypoint
                txt = atwp + " : "

                # Select what to show
                if len(args) == 0:
                    swalt = True
                    swspd = True
                    swat  = True
                else:
                    swalt = args[0].upper() == "ALT"
                    swspd = args[0].upper() in ("SPD","SPEED")
                    swat  = args[0].upper() in ("DO", "STACK")

                    # To be safe show both when we do not know what
                    if not (swalt or swspd or swat):
                        swalt = True
                        swspd = True
                        swat  = True

                # Show altitude
                if swalt:
                    if acrte.wpalt[wpidx] < 0:
                        txt += "-----"

                    elif acrte.wpalt[wpidx] > 4500 * ft:
                        fl = int(round((acrte.wpalt[wpidx] / (100. * ft))))
                        txt += "FL" + str(fl)

                    else:
                        txt += str(int(round(acrte.wpalt[wpidx] / ft)))

                    if swspd:
                        txt += "/"

                # Show speed
                if swspd:
                    if acrte.wpspd[wpidx] < 0:
                        txt += "---"
                    else:
                        txt += str(int(round(acrte.wpspd[wpidx] / kts)))

                # Type
                if swalt and swspd:
                    if acrte.wptype[wpidx] == Route.orig:
                        txt += "[orig]"
                    elif acrte.wptype[wpidx] == Route.dest:
                        txt += "[dest]"

                # Show also stacked commands for when passing this waypoint
                if swat:
                    if len(acrte.wpstack[wpidx])>0:
                        txt = txt+"\nStack:\n"
                        for stackedtxt in acrte.wpstack[wpidx]:
                            txt = txt + stackedtxt + "\n"


                return True, txt

            elif args[0].count("/")==1:
                # Set both alt & speed at this waypoint
                #     KL204 AT LOPIK FL090/250  => acid AT wpinroute alt/spd
                success = True

                # Use parse from stack.py to interpret alt & speed
                alttxt, spdtxt = args[0].split('/')

                # Edit waypoint altitude constraint
                if alttxt.count('-') > 1: # "----" = delete
                    acrte.wpalt[wpidx]  = -999.
                else:
                    try:
                        acrte.wpalt[wpidx] = txt2alt(alttxt)
                        acrte.calcfp()   # Recalculate VNAV axes
                    except ValueError as e:
                        success = False

                # Edit waypoint speed constraint
                if spdtxt.count('-') > 1: # "----" = delete
                    acrte.wpspd[wpidx]  = -999.
                else:
                    try:
                        acrte.wpalt[wpidx] = txt2spd(spdtxt)
                    except ValueError as e:
                        success = False

                if not success:
                    return False,"Could not parse "+args[0]+" as alt / spd"

                # If success: update flight plan and guidance
                acrte.calcfp()
                acrte.direct(acidx, acrte.wpname[acrte.iactwp])

            #acid AT wpinroute ALT/SPD alt/spd
            elif len(args)>=2:
                # KL204 AT LOPIK ALT FL090 => set altitude to be reached at this waypoint in route
                # KL204 AT LOPIK SPD 250 => Set speed at twhich is set at this waypoint
                # KL204 AT LOPIK DO PAN LOPIK => When passing stack command after DO
                # KL204 AT LOPIK STACK PAN LOPIK => AT...STACK synonym for AT...DO
                # KL204 AT LOPIK DO ALT FL240 => => stack "KL204 ALT FL240" => use acid from beginning if omitted as first argument

                swalt = args[0].upper()=="ALT"
                swspd = args[0].upper() in ("SPD","SPEED")
                swat  = args[0].upper() in ("DO","STACK")

                # Use parse from stack.py to interpret alt & speed

                # Edit waypoint altitude constraint
                if swalt:
                    try:
                        acrte.wpalt[wpidx] = txt2alt(args[1])
                    except ValueError as e:
                        return False, e.args[0]

                # Edit waypoint speed constraint
                elif swspd:
                    try:
                        acrte.wpspd[wpidx] = txt2spd(args[1])
                    except ValueError as e:
                        return False, e.args[0]

                # add stack command: args[1] is DO or STACK, args[2:] contains a command
                elif swat:
                    # Check if first argument is missing aircraft id, if so, use this acid

                    # IF command starts with aircraft id, it is not missing
                    cmd = args[1].upper()
                    if not(cmd in bs.traf.id):
                        # Look up arg types
                        try:
                            cmdobj = Command.cmddict[cmd]

                            # Command found, check arguments
                            argtypes = cmdobj.annotations

                            if len(argtypes)>0 and argtypes[0]=="acid" and not (len(args)>2 and args[2].upper() in bs.traf.id):
                                # missing acid, so add ownship acid
                                acrte.wpstack[wpidx].append(acid+" "+" ".join(args[1:]))
                            else:
                                # This command does not need an acid or it is already first argument
                                acrte.wpstack[wpidx].append(" ".join(args[1:]))
                        except:
                            return False, "Stacked command "+cmd+" unknown or syntax error"
                    else:
                        # Command line starts with an aircraft id at the beginning of the command line, stack it
                        acrte.wpstack[wpidx].append(" ".join(args[1:]))

                # Delete a constraint (or both) at this waypoint
                elif args[0]=="DEL" or args[0]=="DELETE" or args[0]=="CLR" or args[0]=="CLEAR" :
                    swalt = args[1].upper()=="ALT"
                    swspd = args[1].upper() in ("SPD","SPEED")
                    swboth  = args[1].upper()=="BOTH"
                    swall   = args[1].upper()=="ALL"

                    if swspd or swboth or swall:
                        acrte.wpspd[wpidx]  = -999.

                    if swalt or swboth or swall:
                        acrte.wpalt[wpidx]  = -999.

                    if swall:
                        acrte.wpstack[wpidx]=[]

                else:
                    return False,"No "+args[0]+" at ",atwp


                # If success: update flight plan and guidance
                acrte.calcfp()
                acrte.direct(acidx, acrte.wpname[acrte.iactwp])

        # Waypoint not found in route
        else:
            return False, atwp + " not found in route " + acid

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
            self.wpturnbank[wpidx] = self.turnbank
            self.wpturnrad[wpidx] = self.turnrad
            self.wpturnspd[wpidx] = self.turnspd
            self.wpturnhdgr[wpidx] = self.turnhdgr
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
            self.wpturnbank.insert(wpidx,self.turnbank)
            self.wpturnrad.insert(wpidx, self.turnrad)
            self.wpturnspd.insert(wpidx, self.turnspd)
            self.wpturnhdgr.insert(wpidx, self.turnhdgr)
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

        #update qdr and "last waypoint switch" in traffic
        if idx>=0:
            bs.traf.actwp.next_qdr[iac] = self.getnextqdr()
            bs.traf.actwp.swlastwp[iac] = (self.iactwp==self.nwp-1)

        # Update waypoints
        if not (wptype == Route.calcwp):
            self.calcfp()

        # Update autopilot settings
        # If we added a waypoint but iactwp is still -1, make it 0
        if wpok and self.iactwp < 0:
            self.iactwp = 0
            
        if wpok and 0 <= self.iactwp < self.nwp:
            self.direct(iac, self.wpname[self.iactwp])

        return idx

    @stack.command(aliases=("DIRECTTO", "DIRTO"))
    @staticmethod
    def direct(acidx: 'acid', wpname: 'wpinroute'):
        """DIRECT acid wpname
        
            Go direct to specified waypoint in route (FMS)"""
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        wpidx = acrte.wpname.index(wpname)

        acrte.iactwp = wpidx
        bs.traf.actwp.lat[acidx]    = acrte.wplat[wpidx]
        bs.traf.actwp.lon[acidx]    = acrte.wplon[wpidx]
        bs.traf.actwp.flyby[acidx]  = acrte.wpflyby[wpidx]
        bs.traf.actwp.flyturn[acidx] = acrte.wpflyturn[wpidx]
        bs.traf.actwp.turnbank[acidx] = acrte.wpturnbank[wpidx]
        bs.traf.actwp.turnrad[acidx] = acrte.wpturnrad[wpidx]
        bs.traf.actwp.turnspd[acidx] = acrte.wpturnspd[wpidx]
        bs.traf.actwp.turnhdgr[acidx] = acrte.wpturnhdgr[wpidx]

        bs.traf.actwp.nextturnlat[acidx], bs.traf.actwp.nextturnlon[acidx], \
        bs.traf.actwp.nextturnspd[acidx], bs.traf.actwp.nextturnrad[acidx], \
        bs.traf.actwp.nextturnhdgr[acidx] , bs.traf.actwp.nextturnidx[acidx]\
            = acrte.getnextturnwp()

        # Determine next turn waypoint data

        # Do calculation for VNAV
        acrte.calcfp()

        bs.traf.actwp.xtoalt[acidx] = acrte.wpxtoalt[wpidx]
        bs.traf.actwp.nextaltco[acidx] = acrte.wptoalt[wpidx]

        bs.traf.actwp.torta[acidx]    = acrte.wptorta[wpidx]    # available for active RTA-guidance
        bs.traf.actwp.xtorta[acidx]  = acrte.wpxtorta[wpidx]  # available for active RTA-guidance

        #VNAV calculations like V/S and speed for RTA
        bs.traf.ap.ComputeVNAV(acidx, acrte.wptoalt[wpidx], acrte.wpxtoalt[wpidx],\
                                    acrte.wptorta[wpidx],acrte.wpxtorta[wpidx])

        # If there is a speed specified, process it
        if acrte.wpspd[wpidx]>0.:
            # Set target speed for autopilot

            if acrte.wpalt[wpidx] < 0.0:
                alt = bs.traf.alt[acidx]
            else:
                alt = acrte.wpalt[wpidx]

            # Check for valid Mach or CAS
            if acrte.wpspd[wpidx] <2.0:
                cas = mach2cas(acrte.wpspd[wpidx], alt)
            else:
                cas = acrte.wpspd[wpidx]

            # Save it for next leg
            bs.traf.actwp.nextspd[acidx] = cas

        # No speed specified for next leg
        else:
            bs.traf.actwp.nextspd[acidx] = -999.


        qdr_,dist_ = geo.qdrdist(bs.traf.lat[acidx], bs.traf.lon[acidx],
                             bs.traf.actwp.lat[acidx], bs.traf.actwp.lon[acidx])

        # Save leg length & direction in actwp data
        bs.traf.actwp.curlegdir[acidx] = qdr_      #[deg]
        bs.traf.actwp.curleglen[acidx] = dist_*nm  #[m]
        
        if acrte.wpflyturn[wpidx] and acrte.wpturnrad[wpidx]>0.: # turn radius specified
            turnrad = acrte.wpturnrad[wpidx]
        # Overwrite is hdgrate  defined
        if acrte.wpflyturn[wpidx] and acrte.wpturnhdgr[wpidx] > 0.: # heading rate specified
            turnrad = bs.traf.tas[acidx]*360./(2*np.pi*acrte.wpturnhdgr[wpidx])
        
        # Update turndist so ComputeVNAV works, is there a next leg direction or not?
        if bs.traf.actwp.next_qdr[acidx] < -900.:
            local_next_qdr = qdr_
        else:
            local_next_qdr = bs.traf.actwp.next_qdr[acidx]


        # check if next waypoint has a speed constraint and give that as TAS
        nextwptspd = bs.traf.actwp.nextspd[acidx]
        nextwpalt = bs.traf.actwp.nextaltco[acidx]

        if nextwptspd > 0 and nextwpalt > 0:
        # Here there is a speed constraint and altitude constraint
            nextwpttas = cas2tas(bs.traf.actwp.nextspd[acidx], bs.traf.actwp.nextaltco[acidx])
        elif nextwptspd > 0:
        # Here there is only a speed constraint
            nextwpttas = cas2tas(bs.traf.actwp.nextspd[acidx], bs.traf.alt[acidx])
        elif nextwpalt > 0:
        # if there is only an altitude constraint
            nextwpttas = cas2tas(bs.traf.cas[acidx], bs.traf.actwp.nextaltco[acidx])
        else:
            nextwpttas = bs.traf.tas[acidx]


        # Calculate turn dist (and radius which we do not use now, but later) now for scalar variable [acidx]
        turndist, turnrad, turnspd, turnbank, turnhdgr = \
            bs.traf.actwp.calcturn(acidx, nextwpttas, qdr_, 
                                    local_next_qdr, bs.traf.actwp.turnbank[acidx], 
                                    bs.traf.actwp.turnrad[acidx],bs.traf.actwp.turnspd[acidx] ,
                                    bs.traf.actwp.turnhdgr[acidx], bs.traf.actwp.flyturn[acidx],
                                    bs.traf.actwp.flyby[acidx])  # update turn distance for VNAV

        # Update turn data
        bs.traf.actwp.turndist[acidx]     = turndist * 1.3
        bs.traf.actwp.turnrad[acidx]      = turnrad
        bs.traf.actwp.turnspd[acidx]      = turnspd
        bs.traf.actwp.turnhdgr[acidx]     = turnhdgr
        bs.traf.ap.turnphi[acidx] = np.deg2rad(turnbank)

        bs.traf.swlnav[acidx] = True
        return True

    @stack.command(name='RTA')
    @staticmethod
    def SetRTA(acidx: 'acid', wpname: 'wpinroute', time: 'time'):  # all arguments of setRTA
        """ RTA acid, wpname, time
        
            Add RTA to waypoint record"""
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        wpidx = acrte.wpname.index(wpname)
        acrte.wprta[wpidx] = time

        # Recompute route and update actwp because of RTA addition
        acrte.direct(acidx, acrte.wpname[acrte.iactwp])

        return True

    @stack.command
    @staticmethod
    def listrte(acidx: 'acid', ipagetxt="0"):
        """ LISTRTE acid, [pagenr]

            Show list of route in window per page of 5 waypoints/"""
        # First get the appropriate ac route
        ipage = int(ipagetxt)
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        if acrte.nwp <= 0:
            return False, "Aircraft has no route."

        for i in range(ipage * 7, ipage * 7 + 7):
            if 0 <= i < acrte.nwp:
                # Name
                if i == acrte.iactwp:
                    txt = "*" + acrte.wpname[i] + " : "
                else:
                    txt = " " + acrte.wpname[i] + " : "

                # Altitude
                if acrte.wpalt[i] < 0:
                    txt += "-----/"

                elif acrte.wpalt[i] > 4500 * ft:
                    fl = int(round((acrte.wpalt[i] / (100. * ft))))
                    txt += "FL" + str(fl) + "/"

                else:
                    txt += str(int(round(acrte.wpalt[i] / ft))) + "/"

                # Speed
                if acrte.wpspd[i] < 0.:
                    txt += "---"
                elif acrte.wpspd[i] > 2.0:
                    txt += str(int(round(acrte.wpspd[i] / kts)))
                else:
                    txt += "M" + str(acrte.wpspd[i])

                # Type: orig, dest, C = flyby, | = flyover, U = flyturn
                if acrte.wptype[i] == Route.orig:
                    txt += "[orig]"
                elif acrte.wptype[i] == Route.dest:
                    txt += "[dest]"
                elif acrte.wpflyturn[i]:
                    txt += "[U]"
                elif acrte.wpflyby[i]:
                    txt += "[C]"
                else: # FLYOVER
                    txt += "[|]"


                # Display message
                stack.echo(txt)

        # Add command for next page to screen command line
        npages = int((acrte.nwp + 6) / 7)
        if ipage + 1 < npages:
            stack.stack(f'INSEDIT LISTRTE {acid},{ipage + 1}')

    def getnextturnwp(self):
        """Give the next turn waypoint data."""
        # Starting point
        wpidx = self.iactwp
        # Find next turn waypoint index
        turnidx_all = np.where(self.wpflyturn)[0]
        argwhere_arr = np.argwhere(turnidx_all>=wpidx)
        if argwhere_arr.size == 0:
            # No turn waypoints, return default values
            return [0., 0., -999., -999., -999, -999.]

        trnidx = turnidx_all[np.argwhere(turnidx_all>=wpidx)[0]][0]

        i = bs.traf.id.index(self.acid)
        # Calculate the turn first. We need to assume  that the aircraft is
        # coming from perfectly on the previous leg.
        qdr, _ = geo.qdrdist(self.wplat[trnidx-1], self.wplon[trnidx-1],
                                self.wplat[trnidx], self.wplon[trnidx])
        if trnidx < len(self.wplat)-1:
            local_next_qdr, _ = geo.qdrdist(self.wplat[trnidx], self.wplon[trnidx],
                                self.wplat[trnidx+1], self.wplon[trnidx+1])
        else:
            local_next_qdr = qdr
            
        turndist, turnrad, turnspd, turnbank, turnhdgr = \
        bs.traf.actwp.calcturn(i, bs.traf.tas[i], qdr, 
                    local_next_qdr, self.wpturnbank[trnidx], 
                    self.wpturnrad[trnidx],self.wpturnspd[trnidx],self.wpturnhdgr[trnidx], True, False)
        
        return [self.wplat[trnidx], self.wplon[trnidx], turnspd, turnrad, turnhdgr, trnidx]

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
                if len(name)>10:
                    if not name[10].isdigit():
                        rwykey = name[8:11]
            # also if it is only RW
            else:
                rwykey = name[7:9]
                if len(name) > 9:
                    if not name[9].isdigit():
                        rwykey = name[7:10]

            # Use this code to look up runway heading
            wphdg = bs.navdb.rwythresholds[name[:4]][rwykey][2]

            # keep constant runway heading
            stack.stack("HDG " + str(self.acid) + " " + str(wphdg))

            # start decelerating
            stack.stack("DELAY " + "10 " + "SPD " + str(self.acid) + " " + "10")

            # delete aircraft
            stack.stack("DELAY " + "42 " + "DEL " + str(self.acid))

            swlastwp = (self.iactwp == self.nwp - 1)

            return self.wplat[self.iactwp],self.wplon[self.iactwp],   \
                           self.wpalt[self.iactwp],self.wpspd[self.iactwp],   \
                           self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp], \
                           self.wpxtorta[self.iactwp], self.wptorta[self.iactwp], \
                           lnavon,self.wpflyby[self.iactwp], \
                           self.wpflyturn[self.iactwp],self.wpturnrad[self.iactwp], \
                           self.wpturnspd[self.iactwp], self.wpturnhdgr[self.iactwp], \
                           self.wpturnbank[self.iactwp], nextqdr, swlastwp

        # Switch LNAV off when last waypoint has been passed
        lnavon = self.iactwp < self.nwp -1

        # if LNAV on: increase counter
        if lnavon:
            self.iactwp += 1

        # Activate switch to indicate that this is the last waypoint (for lenient passing logic in actwp.Reached function)
        swlastwp = (self.iactwp == self.nwp-1)

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
                self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp], \
                self.wpxtorta[self.iactwp], self.wptorta[self.iactwp], \
                lnavon,self.wpflyby[self.iactwp], \
                self.wpflyturn[self.iactwp],self.wpturnrad[self.iactwp], \
                self.wpturnspd[self.iactwp], self.wpturnhdgr[self.iactwp], \
                self.wpturnbank[self.iactwp], nextqdr, swlastwp

    def runactwpstack(self):
        for cmdline in self.wpstack[self.iactwp]:
            stack.stack(cmdline)
            #debug
            # stack.stack("ECHO "+self.acid+" AT "+self.wpname[self.iactwp]+" command issued:"+cmdline)
        return

    @stack.command(aliases=("DELROUTE",))
    @staticmethod
    def delrte(acidx: 'acid' = None):
        """ DELRTE acid
            Delete for this a/c the complete route/dest/orig (FMS)."""
        if acidx is None:
            if bs.traf.ntraf == 0:
                return False, 'No aircraft in simulation'
            if bs.traf.ntraf > 1:
                return False, 'Specify callsign of aircraft to delete route of'
            acidx = 0
        # Simple re-initialize this route as empty
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        acrte.__init__(acid)

        # Also disable LNAV,VNAV if route is deleted
        bs.traf.swlnav[acidx]    = False
        bs.traf.swvnav[acidx]    = False
        bs.traf.swvnavspd[acidx] = False

        return True

    @stack.command(aliases=("DELWP",))
    @staticmethod
    def delwpt(acidx: 'acid', wpname: 'wpinroute'):
        """DELWPT acid,wpname
        
           Delete a waypoint from a route (FMS). """
        # Delete complete route?
        if wpname == "*":
            return Route.delrte(acidx)

        # Look up waypoint
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        try:
            wpidx = acrte.wpname.index(wpname.upper())
        except ValueError:
            return False, "Waypoint " + wpname + " not found"
        # check if active way point is the one being deleted and that it is not the last wpt.
        # If active wpt is deleted then change path of aircraft
        if acrte.iactwp == wpidx and not wpidx == acrte.nwp - 1:
            acrte.direct(acidx, acrte.wpname[wpidx + 1])

        acrte.nwp =acrte.nwp - 1
        del acrte.wpname[wpidx]
        del acrte.wplat[wpidx]
        del acrte.wplon[wpidx]
        del acrte.wpalt[wpidx]
        del acrte.wpspd[wpidx]
        del acrte.wprta[wpidx]
        del acrte.wptype[wpidx]
        del acrte.wpflyby[wpidx]
        del acrte.wpflyturn[wpidx]
        del acrte.wpturnbank[wpidx]
        del acrte.wpturnrad[wpidx]
        del acrte.wpturnspd[wpidx]
        del acrte.wpturnhdgr[wpidx]
        del acrte.wpstack[wpidx]
        if acrte.iactwp > wpidx:
            acrte.iactwp = max(0, acrte.iactwp - 1)

        acrte.iactwp = min(acrte.iactwp, acrte.nwp - 1)

        # If no waypoints left, make sure to disable LNAV/VNAV
        if acrte.nwp==0 and (acidx or acidx==0):
            bs.traf.swlnav[acidx]    =  False
            bs.traf.swvnav[acidx]    =  False
            bs.traf.swvnavspd[acidx] =  False

        return True


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

        # Note: No Top of Descent or Top of Climb can inserted here
        # as this depends on the speed, which might be undefined (often is)
        # Guidance in autpilot.py takes care of ToD and ToC logic while flying using current speed
        # This routine prepares data for this by adding a "ruler" along the flight plan in the form of
        # distance at wp to next altitude constraint (xtoalt), its index ial and the value (toalt)
        # same logic is used for time consarint (requieed time of arrival) RTAs at waypoints

        # Direction to waypoint
        self.nwp = len(self.wpname)

        # Create cleared flight plan calculation table
        self.wpdirfrom   = self.nwp*[0.]  # [deg] Direction of leg laving this waypoint
        self.wpdirto     = self.nwp*[0.]  # [deg] Direction of leg ot this waypoint (if it exists)
        self.wpdistto    = self.nwp*[0.]  # [nm] Distance of leg to this waypoint in nm
        self.wpialt      = self.nwp*[-1]  # wp index of next alttud constraint
        self.wptoalt     = self.nwp*[-999.] # [m] next alt contraint
        self.wpxtoalt    = self.nwp*[1.]  # [m] dist to next alt constraint, default 1.0 to avoid division by zero
        self.wpirta      = self.nwp*[-1]  # wp index of next time constraint
        self.wptorta     = self.nwp*[-999.] # [s] next time constraint
        self.wpxtorta    = self.nwp*[1.]  # [m] dist to next time constraint, default 1.0 to avoid division by zero

        # No waypoints: make empty variables to be safe and return: nothing to do
        if self.nwp==0:
            return

        # Calculate lateral leg data
        # LNAV: Calculate leg distances and directions

        for i in range(0, self.nwp - 1):
            qdr,dist = geo.qdrdist(self.wplat[i]  ,self.wplon[i],
                                self.wplat[i+1],self.wplon[i+1])
            self.wpdirfrom[i] = qdr    # [deg]
            self.wpdistto[i+1]  = dist #[nm]  distto is in nautical miles

        # Also add "from direction" as to directions so no need to shift for actwpdata
        # direction to will be overwritten in actwpdata in case of a direct to
        # Add current pos to first waypoint as default value for direction to 1st waypoint
        iac = bs.traf.id2idx(self.acid)
        qdr,dist = geo.qdrdist(bs.traf.lat[iac],bs.traf.lon[iac],
                               self.wplat[0],self.wplon[0])
        self.wpdirto = [qdr]+self.wpdirfrom[0:-1] #[deg] Direction to waypoints

        # Continue flying in the saem direction
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
        if any(np.array(self.wprta)>=0.0):
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
        wplat  = np.array(self.wplat)
        wplon  = np.array(self.wplon)
        dy = (wplat - bs.traf.lat[i])
        dx = (wplon - bs.traf.lon[i]) * bs.traf.coslat[i]
        dist2 = dx*dx + dy*dy
        # Note: the max() prevents walking back, even in cases when this might be apropriate,
        # such as when previous waypoints have been deleted

        iwpnear = max(self.iactwp, np.argmin(dist2))

        #Unless behind us, next waypoint?
        if iwpnear+1<self.nwp:
            qdr = math.degrees(math.atan2(dx[iwpnear],dy[iwpnear]))
            delhdg = abs(degto180(bs.traf.trk[i]-qdr))

            # we only turn to the first waypoint if we can reach the required
            # heading before reaching the waypoint
            time_turn = max(0.01,bs.traf.tas[i])*math.radians(delhdg)/(g0*math.tan(bs.traf.ap.bankdef[i]))
            time_straight= math.sqrt(dist2[iwpnear])*60.*nm/max(0.01,bs.traf.tas[i])

            if time_turn > time_straight:
                iwpnear += 1

        return iwpnear

    @stack.command
    @staticmethod
    def dumprte(acidx: 'acid'):
        """ DUMPRTE acid

            Write route to output/routelog.txt.
        """
        acid = bs.traf.id[acidx]
        acrte = Route._routes[acid]
        # Open file in append mode, write header
        with open(bs.resource(bs.settings.log_path) / 'routelog.txt', "a") as f:
            f.write("\nRoute "+acid+":\n")
            f.write("(name,type,lat,lon,alt,spd,toalt,xtoalt)  ")
            f.write("type: 0=latlon 1=navdb  2=orig  3=dest  4=calwp\n")

            # write flight plan VNAV data (Lateral is visible on screen)
            for j in range(acrte.nwp):
                f.write( str(( j, acrte.wpname[j], acrte.wptype[j],
                      round(acrte.wplat[j], 4), round(acrte.wplon[j], 4),
                      int(0.5+acrte.wpalt[j]/ft), int(0.5+acrte.wpspd[j]/kts),
                      int(0.5+acrte.wptoalt[j]/ft), round(acrte.wpxtoalt[j]/nm, 3)
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
    
    @stack.command
    @staticmethod
    def cruisespd(acidx: 'acid', spd: 'spd'):
        """Sets the cruise speed of an aircraft in the autopilot. This speed is applied
        after the aircraft exits a turn.
        """
        
        # First, make sure that the velocity is within the performance limits of the aircraft.
        minspd = bs.traf.perf.vmin[acidx]
        maxspd = bs.traf.perf.vmax[acidx]
        spd_to_apply = min(maxspd, max(spd , minspd))
        # Apply the speed
        bs.traf.ap.cruisespd[acidx] = spd_to_apply
        return
