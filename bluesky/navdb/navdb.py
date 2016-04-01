from math import *
import os
import numpy as np

from ..tools.aero import ft
from ..tools.qdr import kwikdist
from ..settings import airport_file


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
        wpswbmp                   : switch indicating whether label bmp is present

        apid                      : list of identifier/short names
        apname                    : long name
        aplat                     : latitude
        aplon                     : longitude
        aptype                    : type of airport (1=large, 2=medium, 3=small)
        apmaxrwy                  : max rwy length in meters
        apco                      : country code
        apswbmp                   : switch indicating whether label bmp is present


    Created by  : Jacco M. Hoekstra (TU Delft)
    """

    def __init__(self,subfolder):
        """read data from subfolder"""

        # Create empty segment indexing lists
        self.wpseg = []
        self.apseg = []
        for lat in range(-90,91):
            self.wpseg.append(361*[[]])
            self.apseg.append(361*[[]])

        #---------- Read waypoints.dat file ----------
        print "Reading waypoints.dat from",subfolder

        # Read data into list of ascii lines
        path = "./data/"+subfolder+"/"
        f = open(path+"waypoints.dat","r")
        lines = f.readlines()
        f.close()

        # Create empty database
        self.wpid    = []              # identifier (string)
        self.wplat   = []              # latitude [deg]
        self.wplon   = []              # longitude [deg]
        self.wpapt   = []              # reference airport {string}
        self.wptype  = []              # type (string)
        self.wpco    = []              # two char country code (string)
        self.wpswbmp = []              # switch indicating whether label bmp is present
        self.wplabel = []              # List to store bitmaps of label

        # Process lines to fill database
        for line in lines:

            # Skip empty lines or comments
            if line.strip()=="":
                continue
            elif line.strip()[0]=="#":
                continue
            
            # Data line => Process fields of this record, separated by a comma
            # Example line:
            # ABARA, , 61.1833, 50.85, UUYY, High and Low Level, RS
            #  [id]    [lat]    [lon]  [airport]  [type] [country code]
            #   0  1     2       3         4        5         6
       
            fields = line.split(",")
            self.wpid.append(fields[0].strip())  # id, no leading or trailing spaces

            self.wplat.append(float(fields[2]))  # latitude [deg]
            self.wplon.append(float(fields[3]))  # longitude [deg]

            self.wpapt.append(fields[4].strip())  # id, no leading or trailing spaces
            self.wptype.append(fields[5].strip().lower())    # type
            self.wpco.append(fields[6].strip())     # country code
            self.wpswbmp.append(False)     # country code

        self.wplat = np.array(self.wplat)
        self.wplon = np.array(self.wplon)
        self.wplabel   = len(self.wpid)*[0]         # list to store bitmaps           

        print "    ",len(self.wpid),"waypoints read."

        #----------  Read airports.dat file ----------
        print "Reading airports.dat from",subfolder

        # Read data into list of ascii lines
        path = "./data/"+subfolder+"/"
        # f = open(path+"airports.dat","r")
        f = open(airport_file, 'r')
        lines = f.readlines()
        f.close()

        # Create empty database
        self.apid      = []              # 4 char identifier (string)
        self.apname    = []              # full name
        self.aplat     = []              # latitude [deg]
        self.aplon     = []              # longitude [deg]
        self.apmaxrwy  = []              # reference airport {string}
        self.aptype    = []              # type (int, 1=large, 2=medium, 3=small)
        self.apco      = []              # two char country code (string)
        self.apswbmp   = []              # switch indicating whether label bmp is present
        self.aplabel   = []              # list to store bitmaps           

        # Process lines to fill database
        types = {'L': 1, 'M': 2, 'S': 3}
        for line in lines:
            # Skip empty lines or comments
            if line.strip()=="":
                continue
            elif line.strip()[0]=="#":
                continue
            
            # Data line => Process fields of this record, separated by a comma
            # Example line:
            # EHAM, SCHIPHOL, 52.309, 4.764, Large, 12467, NL
            #  [id]   [name] [lat]    [lon]  [type] [max rwy length in ft] [country code]
            #   0        1     2        3       4          5                   6
       
            fields = line.split(",")

            # Skip airports without identifier in file and closed airports
            if fields[0].strip()=="" or fields[4].strip() == 'Closed':
                continue

            # print fields[0]
            
            self.apid.append(fields[0].strip())  # id, no leading or trailing spaces
            self.apname.append(fields[1].strip())  # name, no leading or trailing spaces

            self.aplat.append(float(fields[2]))  # latitude [deg]
            self.aplon.append(float(fields[3]))  # longitude [deg]

            self.aptype.append(types[fields[4].strip()[0]])  # large=1, medium=2, small=3

            # Not all airports have rwy length (e.g. heliports)
            try:
                self.apmaxrwy.append(float(fields[5])*ft)  # max rwy ltgh [m]
            except:
                self.apmaxrwy.append(0.0)
            
          
            self.apco.append(fields[6].strip().lower()[:2])     # country code
            self.apswbmp.append(False)     # country code

        self.aplabel   = len(self.apid)*[0]         # list to store bitmaps           

        self.aplat    = np.array(self.aplat)
        self.aplon    = np.array(self.aplon)
        self.apmaxrwy = np.array(self.apmaxrwy)
        self.aptype   = np.array(self.aptype)

        print "    ",len(self.apid),"airports read."

        self.fir = []
        self.firlat0 = []
        self.firlon0 = []
        self.firlat1 = []
        self.firlon1 = []
        
        # Check whether fir subfolder exists
        # try:
        if True:            
            files = os.listdir(path+"fir")
            print "Reading fir subfolder",

            # Get fir names
            for filname in files:
                if filname.count(".txt")>0:

                    firname = filname[:filname.index(".txt")]

                    self.fir.append([firname,[],[]])

                    f = open(path+"fir/"+filname,"r")
                    lines = f.readlines()

                    # Read lines: >N049.28.00.000 E006.20.00.000
                    for line in lines:
                        rec = line.upper().strip()
                        if rec=="":
                            continue
                        latsign = 2*int(line[0]=="N")-1
                        latdeg  = float(line[1:4])
                        latmin  = float(line[5:7])
                        latsec  = float(line[8:14])
                        lat     = latsign*latdeg+latmin/60.+latsec/3600.

                        lonsign = 2*int(line[15]=="E")-1
                        londeg  = float(line[16:19])
                        lonmin  = float(line[20:22])
                        lonsec  = float(line[23:29])
                        lon     = lonsign*londeg+lonmin/60.+lonsec/3600.

                        # For drawing create a line from last lat,lon to current lat,lon
                        if len(self.fir[-1][1])>0:  # skip first lat,lon
                           self.firlat0.append(self.fir[-1][1][-1])                            
                           self.firlon0.append(self.fir[-1][2][-1])
                           self.firlat1.append(lat)                            
                           self.firlon1.append(lon)
                        
                        # Add to FIR record
                        self.fir[-1][1].append(lat)                        
                        self.fir[-1][2].append(lon)

            # Convert lat/lon lines to numpy arrays 
            self.firlat0 = np.array(self.firlat0)
            self.firlat1 = np.array(self.firlat1)
            self.firlon0 = np.array(self.firlon0)
            self.firlon1 = np.array(self.firlon1)

            print len(self.fir)," FIRS read."
            # No fir folders found or error in reading fir files:
        else:
        # except:
            print "No fir folder in",path
        return

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
                dmin = kwikdist(lat,lon,self.wplat[imin],self.wplon[imin])                
                for i in idx[1:]:
                    d = kwikdist(lat,lon,self.wplat[i],self.wplon[i])
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
