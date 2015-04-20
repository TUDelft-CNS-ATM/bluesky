import os
from math import *
import numpy as np
from ..tools.aero import mach2cas, ft, kts, qdrdist, g0, nm, lbs, inch, sqft, rho0, fpm
from ..tools.misc import degto180
from xml.etree import ElementTree


class Waypoint():
    """
    Waypoint class defintion: Waypoint element of FMS route (basic 
        FMS functionality)

    waypoint(name,lat,lon,spd,alt,wptype)  SPD,ALT! as in Legs page on CDU
        spd and alt are constraints, -999 if None (alt=-999)
        
    Created by  : Jacco M. Hoekstra
    """
    
    def __init__(self,name,lat,lon,spd=-999,alt=-999,wptype=0):
        self.name  = name
        self.lat   = lat
        self.lon   = lon
        self.alt   = alt   #[m] negative value means no alt specificied
        self.spd   = spd   #[m/s] negative value means no alt specificied
        self.type  = wptype
        
        self.normal      = 0
        self.origin      = 1
        self.destination = 2
        self.acposition  = 4
        
        return

class Trails():
    """ 
    Traffic trails class definition    : Data for trails

    Methods:
        Trails()            :  constructor

    Members: see create

    Created by  : Jacco M. Hoekstra
    """

    def __init__(self, dttrail=30.):

        self.dt = dttrail  # Resolution of trail pieces in time

        self.tcol0 = 60.  # After how many seconds old colour

        # This list contains some standard colors
        self.colorList = {'BLUE': np.array([0, 0, 255]), \
                          'RED': np.array([255, 0, 0]), \
                          'YELLOW': np.array([255, 255, 0])}
        
        # Set default color to Blue
        self.defcolor = self.colorList['BLUE']

        # Foreground data on line pieces
        self.lat0 = np.array([])
        self.lon0 = np.array([])
        self.lat1 = np.array([])
        self.lon1 = np.array([])
        self.time = np.array([])
        self.col = []
        self.fcol = np.array([])
        self.acid = []

        # background copy of data
        self.bglat0 = np.array([])
        self.bglon0 = np.array([])
        self.bglat1 = np.array([])
        self.bglon1 = np.array([])
        self.bgtime = np.array([])
        self.bgcol = []
        self.bgacid = []

        return

    def update(self, t, aclat, aclon, lastlat, lastlon, lasttim, acid, trailcol):
        """Add linepieces for tr
        ails based on traffic data"""
        
        # Check for update
        delta = t - lasttim
        idxs = np.where(delta > self.dt)[0]

        # Use temporary list for fast append
        lstlat0 = []
        lstlon0 = []
        lstlat1 = []
        lstlon1 = []
        lsttime = []

        # Add all a/c which need the update
        # if len(idxs)>0:
        #     print "len(idxs)=",len(idxs)
        
        for i in idxs:
            # Add to lists
            lstlat0.append(lastlat[i])
            lstlon0.append(lastlon[i])
            lstlat1.append(aclat[i])
            lstlon1.append(aclon[i])
            lsttime.append(t)
            self.acid.append(acid[i])

            if type(self.col) == type(np.array(1)):
                # print type(trailcol[i])
                # print trailcol[i]
                # print "col type: ",type(self.col)
                self.col = self.col.tolist()

            type(self.col)
            self.col.append(trailcol[i])

            # Update aircraft record
            lastlat[i] = aclat[i]
            lastlon[i] = aclon[i]
            lasttim[i] = t

        # Add resulting linepieces
        self.lat0 = np.concatenate((self.lat0, np.array(lstlat0)))
        self.lon0 = np.concatenate((self.lon0, np.array(lstlon0)))
        self.lat1 = np.concatenate((self.lat1, np.array(lstlat1)))
        self.lon1 = np.concatenate((self.lon1, np.array(lstlon1)))
        self.time = np.concatenate((self.time, np.array(lsttime)))

        # Update colours
        self.fcol = (1. - np.minimum(self.tcol0, np.abs(t - self.time)) / self.tcol0)

        return

    def buffer(self):
        """Buffer trails: Move current stack to background"""

        self.bglat0 = np.append(self.bglat0, self.lat0)
        self.bglon0 = np.append(self.bglon0, self.lon0)
        self.bglat1 = np.append(self.bglat1, self.lat1)
        self.bglon1 = np.append(self.bglon1, self.lon1)
        self.bgtime = np.append(self.bgtime, self.time)

        # No color saved: bBackground: always 'old color' self.col0
        if type(self.bgcol) == type(np.array(1)):
            self.bgcol = self.bgcol.tolist()
        if type(self.col) == type(np.array(1)):
            self.col = self.col.tolist()

        self.bgcol = self.bgcol + self.col
        self.bgacid = self.bgacid + self.acid

        self.clearfg()  # Clear foreground trails
        return

    def clearfg(self):  # Foreground
        """Clear trails foreground"""
        self.lat0 = np.array([])
        self.lon0 = np.array([])
        self.lat1 = np.array([])
        self.lon1 = np.array([])
        self.time = np.array([])
        self.col = np.array([])
        self.acid = []
        return 

    def clearbg(self):  # Background
        """Clear trails background"""
        self.bglat0 = np.array([])
        self.bglon0 = np.array([])
        self.bglat1 = np.array([])
        self.bglon1 = np.array([])
        self.bgtime = np.array([])
        self.bgacid = []
        return

    def clear(self):
        """Clear all data, Foreground and background"""
        self.clearfg()
        self.clearbg()
        return

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

        # Current actual waypoint
        self.iactwp = -1

        # Waypoint types:
        self.wplatlon = 0   # lat/lon waypoint        
        self.wpnav    = 1   # VOR/nav database waypoint
        self.orig     = 2   # Origin airport
        self.dest     = 3   # Destination airport
        self.calcwp   = 4   # Calculated waypoint (T/C, T/D, A/C)
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

                # Or add before the first waypoint in route
                else:
                    self.wpname = [name.upper()] + self.wpname
                    self.wptype = [wptype] + self.wptype
                    self.wplat  = [wplat]  + self.wplat
                    self.wplon  = [wplon]  + self.wplon
                    self.wpalt  = [alt]  + self.wpalt
                    self.wpspd  = [spd]  + self.wpspd

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
                       traf.aspd[i] = spd
               vnavok =  True
           else:
               vnavok = False

           qdr, dist = qdrdist(traf.lat[i], traf.lon[i], \
                                 traf.actwplat[i], traf.actwplon[i])

           turnrad = traf.tas[i]*traf.tas[i]/tan(radians(25.)) /g0 /nm # default bank angle 25 deg

                                       
           traf.actwpturn[i] = turnrad*min(4.,abs(tan(0.5*degto180(qdr - \
                        self.wpdirfrom[self.iactwp]))))                    

               
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
               self.wpxtoalt,self.wptoalt, lnavon

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
        epsx = 1.*nm    # [m] Nothing to be done at this hort range
        i = 0
        while i<len(alt)-1:
            if name[i][:2]=="T/":
                continue
            
            dy = alt[i+1]-alt[i]   # alt change (pos = climb)
            dx = x[i]-x[i+1]       # distance (positive)

            dxdes = abs(dy)/desslope
            dxclb = abs(dy)/clbslope

            if dy<eps and  dx + epsx > dxdes:   # insert T/D?

               name.insert(i+1,"T/D")
               alt.insert(i+1,alt[i])
               x.insert(i+1,x[i+1]-dxdes)
               i = i+1
 
            elif dy>eps and  dx + epsx > dxclb:  # insert T/C?                 
               
               name.insert(i+1,"T/C")
               alt.insert(i+1,alt[i+1])
               x.insert(i+1,x[i]+dxclb)
               i = i + 2            
            else:
               i = i + 1
               
        # Now insert T/Cs and T/Ds in actual flight plan
        nvwp = len(alt)
        jwp = self.nwp    # start insert in at zero
        for i in range(nvwp,-1,-1):

            # Copy all new waypoints (which are all named T/C or T/D)
            if name[i][:2]=="T/":

                # Find place in flight plan to insert T/C or T/D
                while dist2go[j]<x[i] and j>1:
                    j=j-1
                    
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
             self.wpdistto[i+1]  = dist

        if self.nwp>1:
            self.wpdirfrom[-1] = self.wpdirfrom[-2]

        # Calclate longitudinal leg data
        # VNAV: calc next altitude constraint: index, altitude and distance to it
        ialt = -1
        toalt = -999.
        xtoalt = 0.
        for i in range(self.nwp-1,-1,-1):
            if self.wptype[i]==self.dest:
                ialt   = i
                toalt  = 0.
                xtoalt = 0.
            elif self.wpalt[i]>=0:
                ialt   = i
                toalt  = self.wpalt[i]
                xtoalt = 0.
            else:
                xtoalt = xtoalt+self.wpdistto[i]
                
            self.wpialt[i] = ialt  
            self.wptoalt[i] = toalt
            self.wpxtoalt[i] = xtoalt
            # print i,self.wpname[i],self.wpxtoalt[i]," to ",self.wptoalt[i]
        return        


class Coefficients:
    """ 
    Coefficient class definition : get aircraft-specific coefficients from database

    Methods:
        coeff() : reading BADA files and store coefficients for all aircraft types

    Created by  : Lisanne Adriaens & Isabel Metz

    This class is based on: 
    EUROCONTROL. User Manual for the Base of Aircraft Data (BADA) Revision 3.12, 
    EEC Technical/Scientific Report No. 14/04/24-44 edition, 2014.

    supported aircraft types (licensed users only):
    A124, A140, A148, A306, A30B, A310, A318, A319, A320, A321, A332, A333, A342, 
    A343, A345, A346, A388, A3ST, AN24, AN28, AN30, AN32, AN38, AT43, AT45, AT72, 
    AT73, AT75, ATP, B190, B350, B462, B463, B703, B712, B722, B732, B733, B734, 
    B735, B736, B737, B738, B739, B742, B743, B744, B748, B752, B753, B762, B763, 
    B764, B772, B773, B77L, B77W, B788, BA11, BE20, BE30, BE40, BE58, BE99, BE9L, 
    C130, C160, C172, C182, C25A, C25B, C25C, C421, C510, C525, C550, C551, C560, 
    C56X, C650, C680, C750, CL60, CRJ1, CRJ2, CRJ9, D228, D328, DA42, DC10, DC87, 
    DC93, DC94, DH8A, DH8C, DH8D, E120, E135, E145, E170, E190, E50P, E55P, EA50, 
    F100, F27, F28, F2TH, F50, F70, F900, FA10, FA20, FA50, FA7X, FGTH, FGTL, FGTN, 
    GL5T, GLEX, GLF5, H25A, H25B, IL76, IL86, IL96, JS32, JS41, L101, LJ35, LJ45, 
    LJ60, MD11, MD82, MD83, MU2, P180, P28A, P28U, P46T, PA27, PA31, PA34, PA44, 
    PA46, PAY2, PAY3, PC12, PRM1, RJ1H, RJ85, SB20, SF34, SH36, SR22, SU95, SW4, 
    T134, T154, T204, TB20, TB21, TBM7, TBM8, YK40, YK42
    """
    def __init__(self):
        # Check opffiles in folder 
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/coefficients/BADA/"))
        self.files = os.listdir(self.path)
        return
        
    def coeff(self):
        # create empty database
        # structure according to BADA OPF files
        self.atype = [] # aircraft type
        self.etype = [] # engine type
        self.engines = [] # engine

        # mass information
        self.mref =  [] # reference mass [t]
        self.mmin =  [] # min mass [t]
        self.mmax =  [] # max mass [t]
        self.mpyld = [] # max payload [t]
        self.gw =    [] # weight gradient on max. alt [ft/kg]

        # flight enveloppe
        self.vmo =  [] # max operating speed [kCAS]
        self.mmo =  [] # max operating mach number 
        self.hmo =  [] # max operating alt [ft]
        self.hmax = [] # max alt at MTOW and ISA [ft]
        self.gt =   [] # temp gradient on max. alt [ft/kg]

        # Surface Area [m^2]
        self.Sref = []

        # Buffet Coefficients
        self.clbo = [] # buffet onset lift coefficient
        self.k =    [] # buffet coefficient
        self.cm16 = [] # CM16
                
        # stall speeds
        self.vsto = [] # stall speed take off [knots] (CAS)
        self.vsic = [] # stall speed initial climb [knots] (CAS)
        self.vscr = [] # stall speed cruise [knots] (CAS)
        self.vsapp = [] # stall speed approach [knots] (CAS)
        self.vsld = [] # stall speed landing [knots] (CAS)

        # minimum speeds
        self.vmto = [] # minimum speed take off [knots] (CAS)
        self.vmic = [] # minimum speed initial climb [knots] (CAS)
        self.vmcr = [] # minimum speed cruise [knots] (CAS)
        self.vmap = [] # minimum speed approach [knots] (CAS)
        self.vmld = [] # minimum speed landing [knots] (CAS)
        self.cvmin = 0 # minimum speed coefficient[-]
        self.cvminto = 0 # minimum speed coefficient [-]
        
        # standard ma speeds
        self.macl = []
        self.macr = []
        self.mades = []
        
        # standard CAS speeds
        self.cascl = []
        self.cascr = []
        self.casdes = []

        # parasitic drag coefficients per phase
        self.cd0to = [] # phase takeoff
        self.cd0ic = [] # phase initial climb
        self.cd0cr = [] # phase cruise
        self.cd0ap = [] # phase approach
        self.cd0ld = [] # phase land
        self.gear = []  # drag due to gear down
        
        # induced drag coefficients per phase
        self.cd2to = [] # phase takeoff
        self.cd2ic = [] # phase initial climb
        self.cd2cr = [] # phase cruise
        self.cd2ap = [] # phase approach
        self.cd2ld = [] # phase land

        self.credj = [] # jet
        self.credt = [] # turbo

        # max climb thrust coefficients
        self.ctcth1 = [] # jet/piston [N], turboprop [ktN]
        self.ctcth2 = [] # [ft]
        self.ctcth3 = [] # jet [1/ft^2], turboprop [N], piston [ktN]

        # 1st and 2nd thrust temp coefficient 
        self.ctct1 = [] # [k]
        self.ctct2 = [] # [1/k]

        # Descent Fuel Flow Coefficients
        # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
        self.ctdesl =  [] # low alt descent thrust coefficient [-]
        self.ctdesh =  [] # high alt descent thrust coefficient [-]
        self.ctdesa =  [] # approach thrust coefficient [-]
        self.ctdesld = [] # landing thrust coefficient [-]

        # transition altitude for calculation of descent thrust
        self.hpdes = [] # [ft]

        # reference speed during descent
        self.vdes = [] # [kCAS]
        self.mdes = [] # [-]

        # Thrust specific fuel consumption coefficients
        self.cf1 = [] # jet [kg/(min*kN)], turboprop [kg/(min*kN*knot)], piston [kg/min]
        self.cf2 = [] # [knots]
        self.cf3 = [] # [kg/min]
        self.cf4 = [] # [ft]
        self.cf5 = [] # [-]

        # ground
        self.tol =  [] # take-off length[m]
        self. ldl = []#landing length[m]
        self.ws =   [] # wingspan [m]
        self.len =  [] # aircraft length[m]

        #  global parameters. 
        # Source: EUROCONTROL (2014). User Manual for the Base of Aircraft Data (BADA) Revision 3.12

        # bank angles
        # self.bank = np.deg2rad(np.array([15,35,35,35,15]))
        
        # minimum speed coefficients
        self.cvmin = 1.3
        self.cvminto = 1.2
        
        # reduced power coefficients
        self.credt = 0.25
        self.credj = 0.15
        self.credp = 0.0
                            
        # read OPF-File
        for f in self.files:
            if ".OPF" in f:
                OPFread = open(self.path + f,'r')
                # Read-in of OPFfiles
                OPFin = OPFread.read()
                # information is given in colums
                OPFin = OPFin.split(' ')

                # get the aircraft type
                OPFname = f[:-6]
                OPFname = OPFname.strip('_')

                # get engine type
                if OPFin[665] == "Jet":
                    etype = 1
                elif OPFin[665] == "Turboprop":
                    etype = 2
                elif OPFin[665] == "Piston":
                    etype = 3
                else:
                    etype = 1
               
                # for some aircraft, the engine name is split up in multiple elements. 
                # Here: Remove non-needed elements to avoid errors when allocating
                # the values in the BADA files to the according parameters
                engine = []
                for i in xrange (len(OPFin)):
                    if OPFname =="C160" or OPFname == "FGTH":
                        continue
                    else:
                        if OPFin[i] =="with":
                            engine.append(OPFin[i+1])
                            i = i+2
                            if OPFin[i] == "engines" or OPFin[i]=="engineswake" or OPFin[i]== "eng." or OPFin[i]=="Engines":
                                break
                            else:
                                while i<len(OPFin):
                                    if OPFin[i]!="engines" and OPFin[i]!="engineswake" and OPFin[i]!= "eng." and OPFin[i]!="Engines":
                                        OPFin.pop(i)
                                    else:
                                        break
                                break
                # only consider numeric values in the remaining document
                for i in range(len(OPFin))[::-1]:
                                   
                    if "E" not in OPFin[i]:
                        OPFin.pop(i) 
                        
                for j in range(len(OPFin))[::-1]:
                    OPFline = OPFin[j]
                    if len(OPFin[j])==0 or OPFline.isalpha() == 1:
                        OPFin.pop(j)

                    # convert all values to floats
                    while j in range(len(OPFin)):
                        try:
                            OPFin[j] = float(OPFin[j])
                            break
                        except ValueError:
                            OPFin.pop(j)

                #add the engine type
                OPFout = OPFin.append(etype)

                # format the result                  
                OPFout = np.asarray(OPFin)
                OPFout = OPFout.astype(float)

                # fill the database                
                self.atype.append(OPFname)
                self.etype.append(OPFout[70])
                self.engines.append(engine)

                # mass information
                self.mref.append(OPFout[0])
                self.mmin.append(OPFout[1])
                self.mmax.append(OPFout[2])
                self.mpyld.append(OPFout[3])
                self.gw.append(OPFout[4])

                # flight enveloppe
                self.vmo.append(OPFout[5])
                self.mmo.append(OPFout[6]) 
                self.hmo.append(OPFout[7])
                self.hmax.append(OPFout[8])
                self.gt.append(OPFout[9])

                # Surface Area [m^2]
                self.Sref.append(OPFout[10])

                # Buffet Coefficients
                self.clbo.append(OPFout[11])
                self.k.append(OPFout[12])
                self.cm16.append(OPFout[13])

                # stall speeds per phase
                self.vsto.append(OPFout[22])
                self.vsic.append(OPFout[18])
                self.vscr.append(OPFout[14])
                self.vsapp.append(OPFout[26])
                self.vsld.append(OPFout[30])

                # minimum speeds
                self.vmto.append(OPFout[22]*self.cvminto)
                self.vmic.append(OPFout[18]*self.cvmin)
                self.vmcr.append(OPFout[14]*self.cvmin)
                self.vmap.append(OPFout[26]*self.cvmin)
                self.vmld.append(OPFout[30]*self.cvmin)

                # parasitic drag coefficients per phase
                self.cd0to.append(OPFout[23])
                self.cd0ic.append(OPFout[19])
                self.cd0cr.append(OPFout[15])
                self.cd0ap.append(OPFout[27])
                self.cd0ld.append(OPFout[31])
                self.gear.append(OPFout[36])
                
                # induced drag coefficients per phase
                self.cd2to.append(OPFout[24])
                self.cd2ic.append(OPFout[20])
                self.cd2cr.append(OPFout[16])
                self.cd2ap.append(OPFout[28])
                self.cd2ld.append(OPFout[32])

                # max climb thrust coefficients
                self.ctcth1.append(OPFout[41]) # jet/piston [N], turboprop [ktN]
                self.ctcth2.append(OPFout[42]) # [ft]
                self.ctcth3.append(OPFout[43]) # jet [1/ft^2], turboprop [N], piston [ktN]

                # 1st and 2nd thrust temp coefficient 
                self.ctct1.append(OPFout[44]) # [k]
                self.ctct2.append(OPFout[45]) # [1/k]

                # Descent Fuel Flow Coefficients
                # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
                self.ctdesl.append(OPFout[46])
                self.ctdesh.append(OPFout[47])
                self.ctdesa.append(OPFout[49])
                self.ctdesld.append(OPFout[50])

                # transition altitude for calculation of descent thrust
                self.hpdes.append(OPFout[48])

                # reference speed during descent
                self.vdes.append(OPFout[51])
                self.mdes.append(OPFout[52])

                # Thrust specific fuel consumption coefficients
                self.cf1.append(OPFout[56])
                self.cf2.append(OPFout[57])
                self.cf3.append(OPFout[58])
                self.cf4.append(OPFout[59])
                self.cf5.append(OPFout[60])

                # ground
                self.tol.append(OPFout[65])
                self. ldl.append(OPFout[66])
                self.ws.append(OPFout[67])
                self.len.append(OPFout[68])         
                OPFread.close()   

            # Airline Procedure Files
            elif ".APF" in f:
                APFread = open(self.path + f,'r')          
            
                for line in APFread.readlines():
                    if not line.startswith("CC") and not line.strip() =="":
                        # whitespaces for splitting the columns
                        content = line.split()

                        # for the moment only values for "average weight" (=mref),
                        # as mach values for all weight classes (low, average, high)
                        # are equal
                        if line.find("AV") != -1:
                            if float(content[5]) < 100:                        
                               self.cascl.append(float(content[4]))
                               self.macl.append(float(content[5]))
                               self.cascr.append(float(content[7]))
                               self.macr.append(float(content[8]))
                               self.mades.append(float(content[9]))  
                               self.casdes.append(float(content[10]))
                            else:
                                self.cascl.append(float(content[5]))
                                self.macl.append(float(content[6]))
                                self.cascr.append(float(content[8]))
                                self.macr.append(float(content[9]))
                                self.mades.append(float(content[10]))
                                self.casdes.append(float(content[11]))
                APFread.close()      

        self.macl = np.array(self.macl)/100
        self.macr = np.array(self.macr)/100
        self.mades = np.array(self.mades)/100
        return


class CoeffBS:
    """ 
    Coefficient class definition : get aircraft-specific coefficients from database
    Created by  : Isabel Metz

    References:

    - D.P. Raymer. Aircraft Design: A Conceptual Approach. AIAA Education Series.
    American Institute of Aeronautics and Astronautics, Inc., Reston, U.S, fifth edition, 2012.
    - R. Babikian. The Historical Fuel Efficiency Characteristics of Regional Aircraft from 
    Technological, Operational, and Cost Perspectives. Master's Thesis, Massachusetts 
    Institute of Technology, Boston, U.S.
    """

    def __init__(self):
        return

    def convert(self, value, unit):
        factors = {'kg': 1., 't':1000, 'lbs': lbs, 'N': 1., 'W': 1, \
                    'm':1.,'km': 1000, 'inch': inch,'ft': ft, \
                    'sqm': 1., 'sqft': sqft, 'sqin': 0.00064516 ,\
                    'm/s': 1., 'km/h': 0.27778, 'kts': kts, 'fpm': fpm, \
                    "kg/s": 1., "kg/m": 1./60., 'mug/J': 0.000001, 'mg/J': 0.001 }
        unit = unit
        try: 
            converted = factors[unit] * float(value)
        except:
            converted = float(value)
            if not self.warned:
                print "Unit missmatch. Could not find ", unit     
                self.warned = True
        return converted 
        

    def coeff(self):

        # aircraft
        self.atype     = [] # aircraft type
        self.j_ac      = [] # list of all jet aircraft
        self.tp_ac     = [] # list of all turboprop aircraft
        
        # engine
        self.etype     = [] # jet / turboprop
        self.engines   = [] # engine types avaliable per aircraft type
        self.j_engines = [] # engine types for jet aircraft
        self.tp_engines= [] # engine types for turboprop aircraft
        self.n_eng     = [] # number of engines
        
        # weights
        self.MTOW      = [] # maximum takeoff weight
       
        # speeds
        self.to_spd    = [] # nominal takeoff speed
        self.ld_spd    = [] # nominal landing speed
        self.max_spd   = [] # maximum CAS
        self.cr_Ma     = [] # nominal cruise Mach at 35000 ft
        self.cr_spd    = [] # cruise speed
        self.max_Ma    = [] # maximum Mach
        
        # limits
        self.vmto   = [] # minimum speed during takeoff
        self.vmld   = [] # minimum speed during landing
        self.clmax_cr = [] # max. cruise lift coefficient
        self.max_alt   = [] # maximum altitude
        
        # dimensions
        #span      = [] # wing span
        self.Sref = [] # reference wing area
        #wet_area  = [] # wetted area
        
        # aerodynamics
        #Cfe       = [] # equivalent skin friction coefficient (Raymer, p.428)
        self.CD0       = [] # parasite drag coefficient
        #oswald    = [] # oswald factor
        self.k         = [] # induced drag factor
        
        # scaling factors for drag (FAA_2005 SAGE)
        # order of flight phases: TO, IC, CR ,AP, LD ,LD gear
        self.d_CD0j = [1.476, 1.143,1.0, 1.957, 3.601, 1.037]
        self.d_kj = [1.01, 1.071, 1.0 ,0.992, 0.932, 1.0]
        self.d_CD0t = [1.220, 1.0, 1.0, 1.279, 1.828, 0.496]
        self.d_kt = [0.948, 1.0, 1.0, 0.94, 0.916, 1.0]
        
        # bank angles per phase. Order: TO, IC, CR, AP, LD. Currently already in CTraffic
        # self.bank = np.deg2rad(np.array([15,35,35,35,15]))

        # flag: did we already warn about invalid input unit?
        self.warned = False
        
        # parse AC files
                
        path = './data/coefficients/BS_aircraft/'
        files = os.listdir(path)
        for file in files:
            acdoc = ElementTree.parse(path + file)

            #actype = doc.find('ac_type')
            self.atype.append(acdoc.find('ac_type').text)

            # engine 
            self.etype.append(int(acdoc.find('engine/eng_type').text))     

            # store jet and turboprop aircraft in seperate lists for accessing specific engine data
            if int(acdoc.find('engine/eng_type').text) ==1:
                self.j_ac.append(acdoc.find('ac_type').text)

            elif int(acdoc.find('engine/eng_type').text) ==2:
                self.tp_ac.append(acdoc.find('ac_type').text)

            self.n_eng.append(float(acdoc.find('engine/num_eng').text))

            engine = []
            for eng in acdoc.findall('engine/eng'):
                engine.append(eng.text)

            # weights
            MTOW = self.convert(acdoc.find('weights/MTOW').text, acdoc.find('weights/MTOW').attrib['unit'])             

            self.MTOW.append(MTOW)   
                
            MLW= self.convert(acdoc.find('weights/MLW').text, acdoc.find('weights/MLW').attrib['unit'])

            # dimensions 
            # wingspan    
            span = self.convert(acdoc.find('dimensions/span').text, acdoc.find('dimensions/span').attrib['unit'])
            # reference surface area   
            S_ref = self.convert(acdoc.find('dimensions/wing_area').text, acdoc.find('dimensions/wing_area').attrib['unit'])
            self.Sref.append(S_ref)   
            
            # wetted area
            S_wet = self.convert(acdoc.find('dimensions/wetted_area').text, acdoc.find('dimensions/wetted_area').attrib['unit'])

            # speeds
            # cruise Mach number
            crma = acdoc.find('speeds/cr_MA')
            if float(crma.text) == 0.0:
                # to be refined
                self.cr_Ma.append(0.8) 
            else:
                self.cr_Ma.append(float(crma.text))

            # cruise TAS
            crspd = acdoc.find('speeds/cr_spd')

            # to be refined
            if float(crspd.text) == 0.0:
                self.cr_spd.append(self.convert(250, 'kts'))
            else: 
                self.cr_spd.append(self.convert(acdoc.find('speeds/cr_spd').text, acdoc.find('speeds/cr_spd').attrib['unit']))

            # limits
            # min takeoff speed
            tospd = acdoc.find('speeds/to_spd')
            if float (tospd.text) == 0.:
                clmax_to = float(acdoc.find('aerodynamics/clmax_to').text)
                self.vmto.append (sqrt((2*g0)/(S_ref*clmax_to))) # influence of current weight and density follows in CTraffic
            else: 
                tospd = self.convert(acdoc.find('speeds/to_spd').text, acdoc.find('speeds/to_spd').attrib['unit'])
                self.vmto.append(tospd/(1.13*sqrt(MTOW/rho0))) # min spd according to CS-/FAR-25.107
            # min ic, cr, ap speed
            clmaxcr = (acdoc.find('aerodynamics/clmax_cr'))
            self.clmax_cr.append(float(clmaxcr.text))
            
            # min landing speed
            ldspd = acdoc.find('speeds/ld_spd')
            if float(ldspd.text) == 0. :                 
                clmax_ld = (acdoc.find('aerodynamics/clmax_ld'))
                self.vmld.append (sqrt((2*g0)/(S_ref*float(clmax_ld.text)))) # influence of current weight and density follows in CTraffic              
            else:
                ldspd = self.convert(acdoc.find('speeds/ld_spd').text, acdoc.find('speeds/ld_spd').attrib['unit'])
                clmax_ld = MLW*g0*2/(rho0*(ldspd**2)*S_ref)
                self.vmld.append(ldspd/(1.23*sqrt(MLW/rho0)))
            # maximum CAS
            maxspd = acdoc.find('limits/max_spd')
            if float(maxspd.text) == 0.0:
                # to be refined
                self.max_spd.append(400.)  
            else:
                self.max_spd.append(self.convert(acdoc.find('limits/max_spd').text, acdoc.find('limits/max_spd').attrib['unit']))
            # maximum Mach
            maxma = acdoc.find('limits/max_MA')
            if float(maxma.text) == 0.0:
                # to be refined
                self.max_Ma.append(0.8)  
            else:
                self.max_Ma.append(float(maxma.text))

                
            # maximum altitude    
            maxalt = acdoc.find('limits/max_alt')
            if float(maxalt.text) == 0.0:
                #to be refined
                self.max_alt.append(11000.)     
            else:
                self.max_alt.append(self.convert(acdoc.find('limits/max_alt').text, acdoc.find('limits/max_alt').attrib['unit'])) 

            # aerodynamics
                
            # parasitic drag - according to Raymer, p. 429
            Cfe = float((acdoc.find('aerodynamics/Cfe').text))
            self.CD0.append (Cfe*S_wet/S_ref) 
            
            # induced drag
            oswald = acdoc.find('aerodynamics/oswald')
            if float(oswald.text) == 0.0:
                # math method according to Obert 2009, p.542: e = 1/(1.02+0.09*pi*AR) combined with Nita 2012, p.2
                self.k.append(1.02/(pi*(span**2/S_ref))+0.009)
            else:
                oswald = float(acdoc.find('aerodynamics/oswald').text)
                self.k.append(1/(pi*oswald*(span**2/S_ref)))
            
            #users = doc.find( 'engine' )
            #for node in users.getiterator():
            #    print node.tag, node.attrib, node.text, node.tail
            
            # to collect avaliable engine types per aircraft
            # 2do!!! access via console so user may choose preferred engine
            # for data file: statistics provided by flightglobal for first choice
            # if not declared differently: first engine is taken!
            self.engines.append(engine)

            if int(acdoc.find('engine/eng_type').text) ==1:
                self.j_engines.append(engine)
                
            elif int(acdoc.find('engine/eng_type').text) ==2:
                self.tp_engines.append(engine)

        # engines
        self.enlist     = [] # list of all engines
        self.jetenlist  = [] # list of all jet engines
        self.propenlist = [] # list of all turbopropengines

        # a. jet aircraft        
        self.rThr       = [] # rated Thrust (one engine)
        self.ffto       = [] # fuel flow takeoff
        self.ffcl       = [] # fuel flow climb
        self.ffcr       = [] # fuel flow cruise
        self.ffid       = [] # fuel flow idle
        self.ffap       = [] # fuel flow approach        
        self.SFC        = [] # specific fuel flow cruise
        
        
        # b. turboprops      
        self.P          = [] # max. power (Turboprops, one engine)
        self.PSFC_TO     = [] # SFC takeoff
        self.PSFC_CR     = [] # SFC cruise

        # parse engine files
        path = os.path.dirname(__file__) + '/../../data/coefficients/BS_engines/'
        files = os.listdir(path)
        for file in files:
            endoc = ElementTree.parse(path + file)
            self.enlist.append(endoc.find('engines/engine').text)

            # thrust
            # a. jet engines            
            if int(endoc.find('engines/eng_type').text) ==1:
                
                # store engine in jet-engine list
                self.jetenlist.append(endoc.find('engines/engine').text)    
                # thrust           
                self.rThr.append(self.convert(endoc.find('engines/Thr').text, endoc.find('engines/Thr').attrib['unit']))
                # bypass ratio    
                BPRc = int(endoc.find('engines/BPR_cat').text)
                # different SFC for different bypass ratios (reference: Raymer, p.36)
                SFC = [14.1, 22.7, 25.5]
                self.SFC.append(SFC[BPRc])
    
                # fuel flow: Takeoff, climb, cruise, approach, idle
                self.ffto.append(self.convert(endoc.find('ff/ff_to').text, endoc.find('ff/ff_to').attrib['unit'])) 
                self.ffcl.append(self.convert(endoc.find('ff/ff_cl').text, endoc.find('ff/ff_cl').attrib['unit'])) 
                self.ffcr.append(self.convert(endoc.find('ff/ff_cr').text, endoc.find('ff/ff_cr').attrib['unit'])) 
                self.ffap.append(self.convert(endoc.find('ff/ff_ap').text, endoc.find('ff/ff_ap').attrib['unit'])) 
                self.ffid.append(self.convert(endoc.find('ff/ff_id').text, endoc.find('ff/ff_id').attrib['unit'])) 

            # b. turboprop engines
            elif int(endoc.find('engines/eng_type').text) ==2:
                
                # store engine in prop-engine list
                self.propenlist.append(endoc.find('engines/engine').text) 
                
                # power
                self.P.append(self.convert(endoc.find('engines/Power').text, endoc.find('engines/Power').attrib['unit']))                 
                # specific fuel consumption: takeoff and cruise   
                PSFC_TO = self.convert(endoc.find('SFC/SFC_TO').text, endoc.find('SFC/SFC_TO').attrib['unit'])
                self.PSFC_TO.append(PSFC_TO) 
                # according to Babikian (function based on PSFC in [mug/J]), input in [kg/J]
                self.PSFC_CR.append(self.convert((0.7675*PSFC_TO*1000000.0 + 23.576), 'mug/J'))
                # print PSFC_TO, self.PSFC_CR
        return