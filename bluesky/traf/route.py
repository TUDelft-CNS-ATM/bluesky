from numpy import *
from ..tools.aero import ft, kts, g0, qdrdist, nm, cas2tas, mach2tas
from ..tools.misc import degto180


class Route():
    """ 
    Route class definition   : Route data for an aircraft (basic FMS functionality)

    addwpt(name,wptype,lat,lon,alt) : Add waypoint (closest to la/lon whene from navdb

    For lat/lon waypoints: use call sign as wpname, number will be added
        
    Created by  : Jacco M. Hoekstra
    """

    def __init__(self,navdb):
        # Add pointer to self navdb object
        self.navdb  = navdb
        self.nwp    = 0

        # Waypoint data
        self.wpname = []
        self.wptype = []
        self.wplat  = []
        self.wplon  = [] 
        self.wpalt  = [] # [m] negative value measn not specified
        self.wpspd  = [] # [m/s] negative value means not specified
        self.wpflyby = [] # Flyby (True)/flyover(False) switch

        # Current actual waypoint
        self.iactwp = -1

        # Waypoint types:
        self.wplatlon = 0   # lat/lon waypoint        
        self.wpnav    = 1   # VOR/nav database waypoint
        self.orig     = 2   # Origin airport
        self.dest     = 3   # Destination airport
        self.calcwp   = 4   # Calculated waypoint (T/C, T/D, A/C)

        self.swflyby = True # Default waypoints are flyby waypoint

        return

    def addwpt(self,traf,iac,name,wptype,lat,lon,alt=-999.,spd=-999.,afterwp=""):
        """Adds waypoint an returns index of waypoint, lat/lon [deg], alt[m]"""
        self.traf = traf  # Traffic object
        self.iac = iac    # a/c to which this route belongs

        # For safety
        self.nwp = len(self.wplat)

        # Be default we trust, distrust needs to be earned
        wpok = True   # switch for waypoint check

        # Select on wptype

        # ORIGIN: Wptype is origin?
        if wptype==self.orig:

             i = self.navdb.getapidx(name.upper().strip())
             wpok = (i >= 0)
             if wpok:
                wplat = self.navdb.aplat[i]
                wplon = self.navdb.aplon[i]            

                # Overwrite existing origin
                if self.nwp>0 and self.wptype[0]==self.orig:
                    self.wpname[0] = name.upper()
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
                if self.iactwp>0:
                    self.iactwp = self.iactwp + 1
             idx = 0

        # DESTINATION: Wptype is destination?
        elif wptype==self.dest:
            i = self.navdb.getapidx(name.upper().strip())
            wpok = (i >= 0)
            if wpok:
                wplat = self.navdb.aplat[i]
                wplon = self.navdb.aplon[i]            

            # Overwrite existing destination
            if wpok and self.nwp>0 and self.wptype[-1]==self.dest:
                self.wpname[-1] = name.upper()
                self.wptype[-1] = wptype
                self.wplat[-1]  = wplat
                self.wplon[-1]  = wplon
                self.wpalt[-1]  = max(0.,alt)  # Use h=0 as default value
                self.wpspd[-1]  = spd
                self.wpflyby[-1] = self.swflyby
                self.nwp = len(self.wpname)            
                idx = self.nwp-1
            
            # Or append to route
            elif wpok:
                self.wpname.append(name.upper())
                self.wptype.append(wptype)
                self.wplat.append(wplat)
                self.wplon.append(wplon)
                self.wpalt.append(max(0.,alt))  # Use h=0 as default value
                self.wpspd.append(spd)
                self.wpflyby.append(self.swflyby)
                self.nwp = len(self.wpname)
                idx = self.nwp-1

                # When only waypoint: adjust pointer to point to destination
                if self.iactwp<0 and self.nwp == 1:
                     self.iactwp = 0
            else:
                idx = -1

        # NORMAL: Wptype is normal waypoint? (lat/lon or nav)
        else:
            # Lat/lon: wpname is then call sign of aircraft: add number 
            if wptype==self.wplatlon:
                newname = name.strip().upper()+"000"
                i     = 0
                while self.wpname.count(newname)>0:
                    i = i + 1               
                    newname = newname[:-3]+str(i).zfill(3)
                wplat = lat
                wplon = lon
                wpok  = True

            # Else make data complete with nav database and closest to given lat,lon
            else:
                newname = name.upper()
                
                i = self.navdb.getwpidx(name.upper().strip(),lat,lon)
                wpok = (i >= 0)
    
                if wpok:

                     newname = name.upper()
                     wplat = self.navdb.wplat[i]
                     wplon = self.navdb.wplon[i]

            # Check if afterwp is specified and found:
            aftwp = afterwp.upper().strip() # Remove space, upper case
            if wpok and afterwp != "" and self.wpname.count(aftwp)>0:
                 wpidx = self.wpname.index(aftwp)+1
                 self.wpname.insert(wpidx,newname)
                 self.wplat.insert(wpidx,wplat)
                 self.wplon.insert(wpidx,wplon)
                 self.wpalt.insert(wpidx,alt)
                 self.wpspd.insert(wpidx,spd)
                 self.wptype.insert(wpidx,wptype)
                 self.wpflyby.insert(wpidx,self.swflyby)
                 if self.iactwp>=wpidx:
                     self.iactwp = self.iactwp + 1

                 idx = wpidx  
            
            # No afterwp: append, just before dest if there is a dest
            elif wpok:

                # Is there a destination?
                if self.nwp>0 and self.wptype[-1]==self.dest:
       
                    # Copy last waypoint and insert before
                    self.wpname.append(self.wpname[-1])
                    self.wplat.append(self.wplat[-1])
                    self.wplon.append(self.wplon[-1])
                    self.wpalt.append(self.wpalt[-1])
                    self.wpspd.append(self.wpspd[-1])
                    self.wptype.append(self.wptype[-1])
                    self.wpflyby.append(self.wpflyby[-1])
                    
                    self.wpname[-2] = newname
                    self.wplat[-2]  = (wplat+90.)%180.-90.
                    self.wplon[-2]  = (wplon+180.)%360.-180.
                    self.wpalt[-2]  = alt
                    self.wpspd[-2]  = spd
                    self.wptype[-2] = wptype
                    idx = self.nwp-2
                
                # Or simply append 
                else:
                    self.wpname.append(newname)
                    self.wplat.append((wplat+90.)%180.-90.)
                    self.wplon.append((wplon+180.)%360.-180.)
                    self.wpalt.append(alt)
                    self.wpspd.append(spd)
                    self.wptype.append(wptype)
                    self.wpflyby.append(flyby)
                    idx = self.nwp-1

        # Update pointers and report whether we are ok
        # debug
        # for i in range(self.nwp):
        #     print i,self.wpname[i]

        if not wpok:
            idx = -1
            if len(self.wplat)==1:
                self.iactwp = 0
        
        # Update waypoints
        if not (wptype == self.calcwp):
            self.calcfp()
            
        return idx

    def direct(self,traf,i,wpnam):
        """Set point to a waypoint by name"""
        name = wpnam.upper().strip()
        if name != "" and self.wpname.count(name)>0:
           wpidx = self.wpname.index(name)
           self.iactwp = wpidx
           traf.actwplat[i] = self.wplat[wpidx]
           traf.actwplon[i] = self.wplon[wpidx]

           if traf.swvnav[i]:
               if self.wpalt[wpidx]:
                    traf.aalt[i] = self.wpalt[wpidx]
               spd = self.wpspd[wpidx]
               if spd>0:
                    if spd<2.0:
                       traf.aspd[i] = mach2cas(spd,traf.alt[i])                            
                    else:    
                       traf.aspd[i] = cas2tas(spd,traf.alt[i])
               vnavok =  True
           else:
               vnavok = False

           qdr, dist = qdrdist(traf.lat[i], traf.lon[i], \
                                 traf.actwplat[i], traf.actwplon[i])

           turnrad = traf.tas[i]*traf.tas[i]/tan(radians(25.)) /g0 /nm # default bank angle 25 deg

                                       
           traf.actwpturn[i] = turnrad*abs(tan(0.5*radians(max(5.,abs(degto180(qdr - \
                        self.wpdirfrom[self.iactwp]))))))                    

               
           return True,vnavok
        else:
           return False,False

    def listrte(self,scr,ipage):
        """LISTRTE command: output route to screen"""
        for i in range(ipage*7,ipage*7+7):
            if 0 <= i <self.nwp:
                # Name
                if i==self.iactwp:
                    txt = "*"+self.wpname[i]+" / "
                else:
                    txt = " "+self.wpname[i]+" / "

                # Altitude
                if self.wpalt[i]<0:
                    txt = txt+"----- / "
                elif self.wpalt[i]>4500*ft:
                    FL = int(round((self.wpalt[i]/(100.*ft))))
                    txt = txt+"FL"+str(FL)+" / "
                else:
                    txt = txt+str(int(round(self.wpalt[i]/ft)))+" / "

                # Speed
                if self.wpspd[i]<0:
                    txt = txt+"---"
                else:
                    txt = txt+str(int(round(self.wpspd[i]/kts)))

                # Type
                if self.wptype[i]==self.orig:
                    txt = txt+ "[orig]"
                elif self.wptype[i]==self.dest:
                    txt = txt+ "[dest]"

                # Display message                
                scr.echo(txt)

    def getnextwp(self):
        """Go to next waypoint and return data"""
        if self.iactwp+1<self.nwp:
            self.iactwp = self.iactwp + 1
            lnavon = True
        else:
            lnavon = False

        return self.wplat[self.iactwp],self.wplon[self.iactwp],   \
               self.wpalt[self.iactwp],self.wpspd[self.iactwp],   \
               self.wpxtoalt[self.iactwp],self.wptoalt[self.iactwp],\
               lnavon,self.wpflyby[self.iactwp]

    def delwpt(self,delwpname):
        """Delete waypoint"""
        
        # Look up waypoint        
        idx = -1
        i = self.nwp
        while idx==-1 and i>0:
            i = i-1
            if self.wpname[i].upper()==delwpname.upper():
                idx = i

        # Delete waypoint        
        if idx>=0:
            self.nwp = self.nwp-1
            del self.wpname[idx]
            del self.wplat[idx]
            del self.wplon[idx]
            del self.wpalt[idx]
            del self.wpspd[idx]
            del self.wptype[idx]

        return idx

    def newcalcfp(self):
        """Do flight plan calculations"""

        # Remove old top of descents and old top of climbs
        while self.wpname.count("T/D")>0:
            self.delwpt("T/D")

        while self.wpname.count("T/C")>0:
            self.delwpt("T/C")

        # Remove old actual position waypoints
        while self.wpname.count("A/C")>0:
            self.delwpt("T/C")

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
            qdr,dist = qdrdist(self.wplat[i],self.wplon[i],    \
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
        self.delwpt("T/D")
        self.delwpt("T/C")

        # Direction to waypoint
        self.nwp = len(self.wpname)

        # Create flight plan calculation table
        self.wpdirfrom   = self.nwp*[0.]
        self.wpdistto    = self.nwp*[0.]
        self.wpialt      = self.nwp*[-1]  
        self.wptoalt     = self.nwp*[0.]
        self.wpxtoalt    = self.nwp*[1.]

        # No waypoints: make empty variables to be safe and return: nothing to do
        if self.nwp==0:
            return

        # Calculate lateral leg data
        # LNAV: Calculate leg distances and directions

        for i in range(0,self.nwp-2):
             qdr,dist = qdrdist(self.wplat[i]  ,self.wplon[i], \
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
            if self.wptype[i]==self.dest or self.wpalt[i] >= 0:
                ialt   = i
                toalt  = self.wpalt[i]
                xtoalt = 0.                # [m]

            # waypoint with no altitude constraint:keep counting
            else:                          
                xtoalt = xtoalt+self.wpdistto[i+1]*nm  # [m] xtoalt is in meters!

            self.wpialt[i] = ialt  
            self.wptoalt[i] = toalt   #[m]
            self.wpxtoalt[i] = xtoalt  #[m]
            # print i,self.wpname[i],self.wpxtoalt[i]," to ",self.wptoalt[i]
        return        
