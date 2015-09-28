from math import *
import numpy as np


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
        self.colorList = {'BLUE': np.array([0, 0, 255]),
                          'RED': np.array([255, 0, 0]),
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
