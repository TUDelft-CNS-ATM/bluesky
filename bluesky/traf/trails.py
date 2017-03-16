from math import *
import numpy as np
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters


class Trails(DynamicArrays):
    """
    Traffic trails class definition    : Data for trails

    Methods:
        Trails()            :  constructor

    Members: see create

    Created by  : Jacco M. Hoekstra
    """

    def __init__(self, traf,dttrail=10.):
        self.active = False  # Wether or not to show trails
        self.dt = dttrail    # Resolution of trail pieces in time

        self.tcol0 = 60.  # After how many seconds old colour
        self.traf = traf

        # This list contains some standard colors
        self.colorList = {'BLUE': np.array([0, 0, 255]),
                          'CYAN': np.array([0,255,255]),
                          'RED' : np.array([255, 0, 0]),
                          'YELLOW': np.array([255, 255, 0])}

        # Set default color to Blue
        self.defcolor = self.colorList['CYAN']

        # Foreground data on line pieces
        self.lat0 = np.array([])
        self.lon0 = np.array([])
        self.lat1 = np.array([])
        self.lon1 = np.array([])
        self.time = np.array([])
        self.col  = []
        self.fcol = np.array([])

        # background copy of data
        self.bglat0 = np.array([])
        self.bglon0 = np.array([])
        self.bglat1 = np.array([])
        self.bglon1 = np.array([])
        self.bgtime = np.array([])
        self.bgcol = []

        with RegisterElementParameters(self):
            self.accolor = []
            self.lastlat = np.array([])
            self.lastlon = np.array([])
            self.lasttim = np.array([])

        self.clearnew()

        return

    def create(self,n=1):
        super(Trails, self).create(n)

        self.accolor[-1] = self.defcolor
        self.lastlat[-1] = self.traf.lat[-1]
        self.lastlon[-1] = self.traf.lon[-1] 
        
    def update(self, t):
        self.acid    = self.traf.id        
        if not self.active:
            self.lastlat = self.traf.lat
            self.lastlon = self.traf.lon
            self.lasttim[:] = t
            return
        """Add linepieces for trails based on traffic data"""

        # Use temporary list/array for fast append
        lstlat0 = []
        lstlon0 = []
        lstlat1 = []
        lstlon1 = []
        lsttime = []

        # Check for update
        delta = t - self.lasttim
        idxs = np.where(delta > self.dt)[0]

        # Add all a/c which need the update
        # if len(idxs)>0:
        #     print "len(idxs)=",len(idxs)
        
        for i in idxs:
            # Add to lists
            lstlat0.append(self.lastlat[i])
            lstlon0.append(self.lastlon[i])
            lstlat1.append(self.traf.lat[i])
            lstlon1.append(self.traf.lon[i])
            lsttime.append(t)

            if isinstance(self.col, np.ndarray):
                # print type(trailcol[i])
                # print trailcol[i]
                # print "col type: ",type(self.col)
                self.col = self.col.tolist()

            type(self.col)
            self.col.append(self.accolor[i])

            # Update aircraft record
            self.lastlat[i] = self.traf.lat[i]
            self.lastlon[i] = self.traf.lon[i]
            self.lasttim[i] = t

        # QtGL send buffer
        self.newlat0.extend(lstlat0)
        self.newlon0.extend(lstlon0)
        self.newlat1.extend(lstlat1)
        self.newlon1.extend(lstlon1)

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

        # No color saved: Background: always 'old color' self.col0
        if isinstance(self.bgcol, np.ndarray):
            self.bgcol = self.bgcol.tolist()
        if isinstance(self.col, np.ndarray):
            self.col = self.col.tolist()

        self.bgcol = self.bgcol + self.col
        self.bgacid = self.bgacid + self.acid

        self.clearfg()  # Clear foreground trails
        return

    def clearnew(self):
        # Clear new lines pipeline used for QtGL
        self.newlat0 = []
        self.newlon0 = []
        self.newlat1 = []
        self.newlon1 = []
      

    def clearfg(self):  # Foreground
        """Clear trails foreground"""
        self.lat0 = np.array([])
        self.lon0 = np.array([])
        self.lat1 = np.array([])
        self.lon1 = np.array([])
        self.time = np.array([])
        self.col = np.array([])
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
        self.lastlon = np.array([])
        self.lastlat = np.array([])
        self.clearfg()
        self.clearbg()
        self.clearnew()
        return

    def setTrails(self, *args):
        """ Set trails on/off, or change trail color of aircraft """
        if type(args[0]) == bool:
            # Set trails on/off
            self.active = args[0]
            if len(args) > 1:
                self.dt = args[1]
            if not self.active:
                self.clear()
        else:
            # Change trail color
            if len(args) < 2 or args[1] not in ["BLUE", "RED", "YELLOW"]:
                return False, "Set aircraft trail color with: TRAIL acid BLUE/RED/YELLOW"
            self.changeTrailColor(args[1], args[0])

    def changeTrailColor(self, color, idx):
        """Change color of aircraft trail"""
        self.accolor[idx] = self.colorList[color]
        return
    
    def reset(self):
        # This ensures that the traffic arrays (which size is dynamic)
        # are all reset as well, so all lat,lon,sdp etc but also objects adsb
        super(Trails, self).reset()
        self.clear()