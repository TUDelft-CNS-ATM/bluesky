from math import *
import numpy as np

from ..tools.loaddata import load_navdata
from ..tools import geo


class Navdatabase:
    """
    Navdatabase class definition : command stack & processing class

    Methods:
        Navdatabase()          :  constructor

        findid(txt,lat,lon)    : find a nav closest to lat,lon


    Members:
        wpid                      : list of identifier/short names
        wpname                    : long name
        wptype                    : type of waypoint (yet unused)
        wplat                     : latitude
        wplon                     : longitude
        wpco                      : country code

        apid                      : list of identifier/short names
        apname                    : long name
        aplat                     : latitude
        aplon                     : longitude
        aptype                    : type of airport (1=large, 2=medium, 3=small)
        apmaxrwy                  : max rwy length in meters
        apco                      : country code


    Created by  : Jacco M. Hoekstra (TU Delft)
    """

    def __init__(self, subfolder):
        """read data from subfolder"""

        # Create empty segment indexing lists
        self.wpseg = []
        self.apseg = []
        for lat in range(-90, 91):
            self.wpseg.append(361 * [[]])
            self.apseg.append(361 * [[]])

        wptdata, aptdata, firdata, rwythresholds = load_navdata()
        self.wpid      = wptdata['wpid']  # identifier (string)
        self.wplat     = wptdata['wplat']  # latitude [deg]
        self.wplon     = wptdata['wplon']  # longitude [deg]
        self.wpapt     = wptdata['wpapt']  # reference airport {string}
        self.wptype    = wptdata['wptype']  # type (string)
        self.wpco      = wptdata['wpco']  # two char country code (string)

        # Create empty database
        self.apid      = aptdata['apid']      # 4 char identifier (string)
        self.apname    = aptdata['apname']    # full name
        self.aplat     = aptdata['aplat']     # latitude [deg]
        self.aplon     = aptdata['aplon']     # longitude [deg]
        self.apmaxrwy  = aptdata['apmaxrwy']  # reference airport {string}
        self.aptype    = aptdata['aptype']    # type (int, 1=large, 2=medium, 3=small)
        self.apco      = aptdata['apco']      # two char country code (string)

        self.fir       = firdata['fir']
        self.firlat0   = firdata['firlat0']
        self.firlon0   = firdata['firlon0']
        self.firlat1   = firdata['firlat1']
        self.firlon1   = firdata['firlon1']

        self.rwythresholds = rwythresholds

    def getwpidx(self,txt,lat=999999.,lon=999999):
        """Get waypoint index to access data"""
        name = txt.upper()
        try:
            i = self.wpid.index(name)
        except:
            return -1
       
        # if no pos is specified, get first occurence
        if not lat<99999.:
            return i

        # If pos is specified check for more and return closest
        else:    
            idx = []
            idx.append(i)
            found = True
            while i<len(self.wpid)-1 and found:
                try:
                    i = self.wpid.index(name,i+1)
                    idx.append(i)
                except:
                    found = False
            if len(idx)==1:
                return idx[0]
            else:
                imin = idx[0]
                dmin = geo.kwikdist(lat,lon,self.wplat[imin],self.wplon[imin])                
                for i in idx[1:]:
                    d = geo.kwikdist(lat,lon,self.wplat[i],self.wplon[i])
                    if d<dmin:
                        imin = i
                        dmin = d
                return imin

    def getapidx(self,txt):
        """Get waypoint index to access data"""
        try:
            return self.apid.index(txt.upper())
        except:
            return -1

    def getinear(self,wlat,wlon,lat,lon): # lat,lon in degrees
        # t0 = time.clock()
        f = cos(radians(lat))
        dlat = (wlat-lat+180.)%360.-180.
        dlon = f*((wlon-lon+180.)%360.-180.)
        d2 = dlat*dlat+dlon*dlon
        idx = np.argmin(d2)
        # dt = time.clock()-t0
        # print dt
        return idx    

    def getwpinear(self,lat,lon): # lat,lon in degrees
        """Get closest waypoint index"""
        return self.getinear(self.wplat,self.wplon,lat,lon)  

    def getapinear(self,lat,lon): # lat,lon in degrees
        """Get closest airport index"""
        return self.getinear(self.aplat,self.aplon,lat,lon)  

    def getinside(self,wlat,wlon,lat0,lat1,lon0,lon1):
        """Get indices inside given box"""
        # t0 = time.clock()
        if lat0 < lat1:
            arr = np.where((wlat>lat0)*(wlat<lat1)*(wlon>lon0)*(wlon<lon1))
        else:
            arr = np.where((wlat>lat1)+(wlat<lat0)*(wlon>lon0)*(wlon<lon1))

        # dt = time.clock()-t0
        # print dt
        return list(arr[0])# Get indices        

    def getwpinside(self,lat0,lat1,lon0,lon1):
        """Get waypoint indices inside box"""
        return self.getinside(self.wplat,self.wplon,lat0,lat1,lon0,lon1)        

    def getapinside(self,lat0,lat1,lon0,lon1):
        """Get airport indicex inside box"""
        return self.getinside(self.aplat,self.aplon,lat0,lat1,lon0,lon1)        
        
    # returns all runways of given airport  
    def listrwys(self,ICAO):
        return True, str(self.rwythresholds[ICAO].keys())
