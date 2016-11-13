from numpy import *
from ..tools import geo
from ..tools.aero import ft, kts, g0, nm, cas2tas, mach2cas
from ..tools.misc import degto180
from ..tools.position import txt2pos

class Route():
    """
    Route class definition   : Route data for an aircraft (basic FMS functionality)

    addwpt(name,wptype,lat,lon,alt) : Add waypoint (closest to la/lon whene from navdb

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

    def __init__(self, navdb):
        
        # Save a local pointer to the navigation database object navdb
        self.navdb  = navdb
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

        return

    def addwptStack(self, traf, idx, *args): # args: all arguments of addwpt
        "ADDWPT acid, (wpname/lat,lon),[alt],[spd],[afterwp]"

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
        name = args[0]

        success,posobj = txt2pos(name,traf,self.navdb,traf.lat[idx],traf.lon[idx])
        if success:        
            
            lat      = posobj.lat
            lon      = posobj.lon
            
            if posobj.type== "nav" or posobj.type== "apt":
                wptype = self.wpnav
    
            elif posobj.type == "rwy":
                wptype  = self.runway
    
            else: # treat as lat/lon
                name   = traf.id[idx]            
                wptype   = self.wplatlon
    
            # Default altitude, speed and afterwp if not given
            alt     = -999.  if len(args) < 2 else args[1]
            spd     = -999.  if len(args) < 3 else args[2]
            afterwp = ""     if len(args) < 4 else args[3]
    
            #Catch empty arguments (None)
            if alt=="" or alt==None:
                alt = -999
            
            if spd=="" or spd==None:
                spd = -999
            
            if afterwp==None:
                afterwp = ""
    
            # Add waypoint
            wpidx = self.addwpt(traf, idx, name, wptype, lat, lon, alt, spd, afterwp)
            
            # Check for success by checking insetred locaiton in flight plan >= 0
            if wpidx < 0:
                return False, "Waypoint " + name + " not added."
    
            # chekc for presence of orig/dest
            norig = int(traf.ap.orig[idx] != "")
            ndest = int(traf.ap.dest[idx] != "")
    
            # Check whether this is first 'real' wayppint (not orig & dest), 
            # And if so, make active
            if self.nwp - norig - ndest == 1:  # first waypoint: make active
                self.direct(traf, idx, self.wpname[norig])  # 0 if no orig
                traf.swlnav[idx] = True
    
            if afterwp and self.wpname.count(afterwp) == 0:
                return True, "Waypoint " + afterwp + " not found" + \
                    "waypoint added at end of route"
            else:
                return True

        else:
             return False,"Waypoint "+name+" not found."


    def addwpt(self,traf,iac,name,wptype,lat,lon,alt=-999.,spd=-999.,afterwp=""):
        """Adds waypoint an returns index of waypoint, lat/lon [deg], alt[m]"""
        self.traf = traf  # Traffic object
        self.iac = iac    # a/c to which this route belongs
        # For safety
        self.nwp = len(self.wplat)

        # Be default we trust, distrust needs to be earned
        wpok = True   # switch for waypoint check

        # Check if name already exists, if so add integer 01, 02, 03 etc.
        appi    = 0 # appended integer to name starts at zero (=nothing)
        wprtename = name.upper()  # wp name for in route
        while self.wpname.count(wprtename)>0:
            appi = appi+1
            wprtename = name.upper()+"%02d"%appi

        # Select on wptype
        # ORIGIN: Wptype is origin?
        if wptype == self.orig:

            if not (name == traf.id[iac] + "ORIG"):   # published identifier
                i = self.navdb.getaptidx(name.upper().strip())
                wpok = (i >= 0)
                if wpok:
                    wplat = self.navdb.aptlat[i]
                    wplon = self.navdb.aptlon[i]
            else:                                 # lat/lon type
                wplat = lat
                wplon = lon
                wpok  = True

            if wpok:
                # Overwrite existing origin
                if self.nwp > 0 and self.wptype[0] == self.orig:
                    self.wpname[0] = wprtename
                    self.wptype[0] = wptype
                    self.wplat[0]  = wplat
                    self.wplon[0]  = wplon
                    self.wpalt[0]  = alt
                    self.wpspd[0]  = spd
                    self.wpflyby[0] = self.swflyby

                # Or add before the first waypoint in route
                else:
                    self.wpname = [name.upper()] + self.wpname
                    self.wptype = [wptype] + self.wptype
                    self.wplat  = [wplat]  + self.wplat
                    self.wplon  = [wplon]  + self.wplon
                    self.wpalt  = [alt]  + self.wpalt
                    self.wpspd  = [spd]  + self.wpspd
                    self.wpflyby = [self.swflyby] + self.wpflyby

                self.nwp    = self.nwp + 1
                if self.iactwp > 0:
                    self.iactwp = self.iactwp + 1
                    
            idx = 0

        # DESTINATION: Wptype is destination?

        elif wptype == self.dest:

            if not (name == traf.id[iac] + "DEST"):   # published identifier
                i = self.navdb.getaptidx(name.upper().strip())
                wpok = (i >= 0)
                if wpok:
                    wplat = self.navdb.aptlat[i]
                    wplon = self.navdb.aptlon[i]
            else:                                 # lat/lon type
                wplat = lat
                wplon = lon
                wpok  = True

            # Overwrite existing destination
            if wpok and self.nwp > 0 and self.wptype[-1] == self.dest:
                self.wpname[-1] = name.upper()
                self.wptype[-1] = wptype
                self.wplat[-1]  = wplat
                self.wplon[-1]  = wplon
                self.wpalt[-1]  = max(0., alt)  # Use h=0 as default value
                self.wpspd[-1]  = spd
                self.wpflyby[-1] = self.swflyby
                self.nwp = len(self.wpname)
                idx = self.nwp - 1

            # Or append to route
            elif wpok:
                self.wpname.append(name.upper())
                self.wptype.append(wptype)
                self.wplat.append(wplat)
                self.wplon.append(wplon)
                self.wpalt.append(max(0., alt))  # Use h=0 as default value
                self.wpspd.append(spd)
                self.wpflyby.append(self.swflyby)
                self.nwp = len(self.wpname)
                idx = self.nwp - 1

                # When only waypoint: adjust pointer to point to destination
                if self.iactwp < 0 and self.nwp == 1:
                    self.iactwp = 0
            else:
                idx = -1

        # NORMAL: Wptype is normal waypoint? (lat/lon or nav)
        else:
            # Lat/lon: wpname is then call sign of aircraft: add number
            if wptype == self.wplatlon:
                newname = name.strip().upper() + "000"
                i     = 0
                while self.wpname.count(newname) > 0:
                    i = i + 1
                    newname = newname[:-3] + str(i).zfill(3)
                wplat = lat
                wplon = lon
                wpok  = True

            # Else make data complete with nav database and closest to given lat,lon
            else: # so wptypewpnav
                newname = wprtename

                if wptype == self.runway:
                    wplat = lat
                    wplon = lon
                    wpok  = True

                else:
                    i = self.navdb.getwpidx(name.upper().strip(), lat, lon)
                    wpok = (i >= 0)
    
                    if wpok:
                        newname = wprtename
                        wplat = self.navdb.wplat[i]
                        wplon = self.navdb.wplon[i]
                    else:
                        i = self.navdb.getaptidx(name.upper().strip())
                        wpok = (i >= 0)
                        if wpok:
                            newname = wprtename
                            wplat = self.navdb.aptlat[i]
                            wplon = self.navdb.aptlon[i]
                        else:
                            newname = wprtename
                            wplat = lat
                            wplon = lon
                        

            # Check if afterwp is specified and found:
            aftwp = afterwp.upper().strip()  # Remove space, upper case
            if wpok:  
            
                if afterwp != "" and self.wpname.count(aftwp) > 0:
                    wpidx = self.wpname.index(aftwp) + 1
                    self.wpname.insert(wpidx, newname)
                    self.wplat.insert(wpidx, wplat)
                    self.wplon.insert(wpidx, wplon)
                    self.wpalt.insert(wpidx, alt)
                    self.wpspd.insert(wpidx, spd)
                    self.wptype.insert(wpidx, wptype)
                    self.wpflyby.insert(wpidx, self.swflyby)
                    if self.iactwp >= wpidx:
                        self.iactwp = self.iactwp + 1
    
                    idx = wpidx
    
                # No afterwp: append, just before dest if there is a dest
                else:

                # Is there a destination?
                    if self.nwp > 0 and self.wptype[-1] == self.dest:
    
                        # Copy last waypoint and insert before
                        self.wpname.append(self.wpname[-1])
                        self.wplat.append(self.wplat[-1])
                        self.wplon.append(self.wplon[-1])
                        self.wpalt.append(self.wpalt[-1])
                        self.wpspd.append(self.wpspd[-1])
                        self.wptype.append(self.wptype[-1])
                        self.wpflyby.append(self.wpflyby[-1])
    
                        self.wpname[-2] = newname
                        self.wplat[-2]  = (wplat + 90.) % 180. - 90.
                        self.wplon[-2]  = (wplon + 180.) % 360. - 180.
                        self.wpalt[-2]  = alt
                        self.wpspd[-2]  = spd
                        self.wptype[-2] = wptype

                        # Update pointers and report whether we are ok
                        self.nwp = len(self.wplat)
                        idx = self.nwp - 2
                    # Or simply append
                    else:
                        self.wpname.append(newname)
                        self.wplat.append((wplat + 90.) % 180. - 90.)
                        self.wplon.append((wplon + 180.) % 360. - 180.)
                        self.wpalt.append(alt)
                        self.wpspd.append(spd)
                        self.wptype.append(wptype)
                        self.wpflyby.append(self.swflyby)
 
                        # Update pointers and report whether we are ok
                        self.nwp = len(self.wplat) 
                        idx = self.nwp-1 
            else:
                idx = -1
                if len(self.wplat) == 1:
                    self.iactwp = 0

            #update qdr in traffic
            traf.actwp.next_qdr[iac] = self.getnextqdr()        
            
        # Update waypoints
        if not (wptype == self.calcwp):
            self.calcfp()

        # Update autopilot settings
        if wpok and self.iactwp >= 0 and self.iactwp < self.nwp:
            self.direct(traf, iac, self.wpname[self.iactwp])


        return idx

    def direct(self, traf, i, wpnam):
        """Set active point to a waypoint by name"""
        name = wpnam.upper().strip()
        if name != "" and self.wpname.count(name) > 0:
            wpidx = self.wpname.index(name)
            self.iactwp = wpidx
            traf.actwp.lat[i] = self.wplat[wpidx]
            traf.actwp.lon[i] = self.wplon[wpidx]

            self.calcfp()
            traf.ap.ComputeVNAV(i,self.wptoalt[wpidx],self.wpxtoalt[wpidx])
            if traf.swvnav[i]:
#            if True:
#                # Set target altitude for autopilot
#                if self.wptoalt[wpidx] > 0:
#
#                    if traf.alt[i] < self.wptoalt[i]-10.*ft:
#                        traf.actwp.alt[i] = self.wptoalt[wpidx]
#                        traf.ap.dist2vs[i] = 9999.
#                    else:
#                    
#                        steepness = 3000.*ft/(10.*nm)
#                        traf.actwp.alt[i] = self.wptoalt[wpidx] + self.wpxtoalt[wpidx]*steepness
#                        delalt = traf.alt[i] - traf.actwp.alt[i]
#                        traf.ap.dist2vs[i] = steepness*delalt

                # Set target speed for autopilot
                spd = self.wpspd[wpidx]
                if spd > 0:
                    if spd < 2.0:
                        traf.aspd[i] = mach2cas(spd, traf.alt[i])
                    else:
                        traf.aspd[i] = cas2tas(spd, traf.alt[i]) # or is '= spd'

            qdr, dist = geo.qdrdist(traf.lat[i], traf.lon[i],
                                traf.actwp.lat[i], traf.actwp.lon[i])

            turnrad = traf.tas[i]*traf.tas[i]/tan(radians(25.)) / g0 / nm  # default bank angle 25 deg

            traf.actwp.turndist[i] = turnrad*abs(tan(0.5*radians(max(5., abs(degto180(qdr -
                        self.wpdirfrom[self.iactwp]))))))

            traf.swlnav[i] = True
            return True
        else:
            return False, "Waypoint " + wpnam + " not found"

    def listrte(self, scr, idx, traf, ipage=0):
        """LISTRTE command: output route to screen"""
        if self.nwp <= 0:
            return False, "Aircraft has no route."

        if idx<0:
            return False, "Aircraft id not found."

        for i in range(ipage * 7, ipage * 7 + 7):
            if 0 <= i < self.nwp:
                # Name
                if i == self.iactwp:
                    txt = "*" + self.wpname[i] + " / "
                else:
                    txt = " " + self.wpname[i] + " / "

                # Altitude
                if self.wpalt[i] < 0:
                    txt = txt+"----- / "
                    
                elif self.wpalt[i] > 4500 * ft:
                    FL = int(round((self.wpalt[i]/(100.*ft))))
                    txt = txt+"FL"+str(FL)+" / "
                    
                else:
                    txt = txt+str(int(round(self.wpalt[i] / ft))) + " / "

                # Speed
                if self.wpspd[i] < 0:
                    txt = txt+"---"
                else:
                    txt = txt+str(int(round(self.wpspd[i] / kts)))

                # Type
                if self.wptype[i] == self.orig:
                    txt = txt + "[orig]"
                elif self.wptype[i] == self.dest:
                    txt = txt + "[dest]"

                # Display message
                scr.echo(txt)

        # Add command for next page to screen command line
        npages = int((self.nwp + 6) / 7)
        if ipage + 1 < npages:
            scr.cmdline("LISTRTE "+traf.id[idx]+","+str(ipage+1))

    def getnextwp(self):
        """Go to next waypoint and return data"""
        lnavon = self.iactwp +1 < self.nwp
        if lnavon:
            self.iactwp = self.iactwp + 1
            lnavon = True
        else:
            lnavon = False
            
        nextqdr= self.getnextqdr()                                                          

        return self.wplat[self.iactwp],self.wplon[self.iactwp],   \
               self.wpalt[self.iactwp],self.wpspd[self.iactwp],   \
               self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp],\
               lnavon,self.wpflyby[self.iactwp], nextqdr

    def delwpt(self, delwpname):
        """Delete waypoint"""

        # Look up waypoint
        idx = -1
        i = len(self.wpname)
        while idx == -1 and i > 1:
            i = i-1
            if self.wpname[i].upper() == delwpname.upper():
                idx = i

        # Delete waypoint
        if idx == -1:
            return False, "Waypoint " + delwpname + " not found"

        self.nwp = self.nwp-1
        del self.wpname[idx]
        del self.wplat[idx]
        del self.wplon[idx]
        del self.wpalt[idx]
        del self.wpspd[idx]
        del self.wptype[idx]
        if self.iactwp > idx:
            self.iactwp = max(0, self.iactwp - 1)

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
        self.insertcalcwp(self,idx,"A/C")
        self.wplat[idx] = self.traf.lat[self.iac] # deg
        self.wplon[idx] = self.traf.lon[self.iac] # deg
        self.wpalt[idx] = self.traf.alt[self.iac] # m
        self.wpspd[idx] = self.traf.tas[self.iac] # m/s

        # Calculate distance to last waypoint in route
        dist2go = []
        nwp = len(wpname)
        dist2go = [0.0]
        for i in range(nwp-2,-1,-1):
            qdr,dist = geo.qdrdist(self.wplat[i],self.wplon[i],    \
                        self.wplat[i+1],self.wplon[i+1])
            dist2go = [dist2go[i+1]+dist]+dist2go

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

        crzalt = self.traf.crzalt[self.iac]
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


        # Now find level segments in climb and descent
        nvwp = len(alt)
                    
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
               i = i+1
 
            elif dy>epsh and  dx + epsx > dxclb:  # insert T/C?                 
               
               name.insert(i+1,"T/C")
               alt.insert(i+1,alt[i+1])
               x.insert(i+1,x[i]+dxclb)
               i = i + 2            
            else:
               i = i + 1
               
        # Now insert T/Cs and T/Ds in actual flight plan
        nvwp = len(alt)
        for i in range(nvwp,-1,-1):

            # Copy all new waypoints (which are all named T/C or T/D)
            if name[i][:2]=="T/":

                # Find place in flight plan to insert T/C or T/D
                while dist2go[j]<x[i] and j>1:
                    j=j-1

                # Interpolation factor for position on leg                    
                f   = (x[i]-dist2go[j+1])/(dist2go[j]-dist2go[j+1])

                lat = f*self.wplat[j]+(1.-f)*wplat[j+1]
                lon = f*self.wplon[j]+(1.-f)*wplon[j+1]

                self.wpname.insert(j,name[i])
                self.wptype.insert(j,self.calcwp)
                self.wplat.insert(j,lat)
                self.wplon.insert(j,lon)
                self.wpalt.insert(j,alt[i])
                self.wpspd.insert(j,-999.)

        return        

    def insertcalcwp(self,i,name):
        """Insert empty wp with no attributes at location i"""

        self.wpname.insert(i,name)
        self.wplat.insert(i,0.)
        self.wplon.insert(i,0.)
        self.wpalt.insert(i,-999.)
        self.wpspd.insert(i,-999.)
        self.wptype.insert(i,self.calcwp)
        return          


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

        for i in range(0,self.nwp-2):
             qdr,dist = geo.qdrdist(self.wplat[i]  ,self.wplon[i], \
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
        for i in range(self.nwp-2,-1,-1):

            # waypoint with altitude constraint (dest of al specified)        
            if self.wptype[i]==self.dest:
                ialt   = i
                toalt  = 0.
                xtoalt = 0.                # [m]

            elif self.wpalt[i] >= 0:
                ialt   = i
                toalt  = self.wpalt[i]
                xtoalt = 0.                # [m]

            # waypoint with no altitude constraint:keep counting
            else:                          
                xtoalt = xtoalt+self.wpdistto[i+1]*nm  # [m] xtoalt is in meters!

            self.wpialt[i] = ialt  
            self.wptoalt[i] = toalt   #[m]
            self.wpxtoalt[i] = xtoalt  #[m]
        return        

    def findact(self,traf,i):
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
        dy = wplat - traf.lat[i] 
        dx = (wplon - traf.lon[i]) * traf.coslat[i]
        dist2 = dx*dx + dy*dy            
        iwpnear = argmin(dist2)

        #Unless behind us, next waypoint?
        if iwpnear+1<self.nwp:
            qdr = arctan2(dx[iwpnear],dy[iwpnear])
            delhdg = abs(degto180(traf.trk[i]-qdr))            
            
            # we only turn to the first waypoint if we can reach the required
            # heading before reaching the waypoint
            time_turn = max(0.01,traf.tas[i])*radians(delhdg)/(g0*tan(traf.bank[i]))
            time_straight= dist2[iwpnear]*nm/max(0.01,traf.tas[i])
            
            if time_turn >time_straight:
                iwpnear = iwpnear+1         

        return iwpnear

    def dumpRoute(self, traf, idx):
        acid = traf.id[idx]
        # Open file in append mode, write header
        with open("./data/output/routelog.txt", "a") as f:
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
        if self.iactwp+1<self.nwp:
            nextqdr, dist = geo.qdrdist(\
                        self.wplat[self.iactwp],  self.wplon[self.iactwp],\
                        self.wplat[self.iactwp+1],self.wplon[self.iactwp+1]) 
        else:
            nextqdr = -999. 
        return nextqdr

