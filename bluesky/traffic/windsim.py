''' Simulate wind in BlueSky. '''
from numpy import arctan2,degrees,array,sqrt # to allow arrays, their functions and types

from bluesky.tools.aero import kts
from bluesky.core import Entity
from .windfield import Windfield


class WindSim(Entity, Windfield, replaceable=True):
    def add(self, *arg):

        lat = arg[0]
        lon = arg[1]
        winddata = arg[2:]

        ndata = len(winddata)

        # No altitude or just one: same wind for all altitudes at this position
        if ndata == 3 or (ndata == 4 and winddata[0] is None): # only one point, ignore altitude
            if winddata[1] is None or winddata[2] is None:
               return False, "Wind direction and speed needed."

            self.addpoint(lat,lon,winddata[1],winddata[2]*kts)

        # More than one altitude is given
        elif ndata > 3:
            windarr = array(winddata)
            dirarr = windarr[1::3]
            spdarr = windarr[2::3] * kts
            altarr = windarr[0::3]

            self.addpoint(lat,lon,dirarr,spdarr,altarr)

        elif winddata.count("DEL") > 0:
            self.clear()

        else:# Something is wrong
            return False, "Winddata not recognized"

        return True

    def get(self, lat, lon, alt=None):
        """ Get wind vector at gioven position (and optioanlly altitude)"""
        vn,ve = self.getdata(lat,lon,alt)

        wdir = (degrees(arctan2(ve,vn)) + 180.) % 360.
        wspd = sqrt(vn * vn + ve * ve)

        txt  = "WIND AT %.5f, %.5f: %03d/%d" % (lat,lon,round(wdir),round(wspd/kts))

        return True, txt
