''' Simulate wind in BlueSky. '''
from numpy import arctan2,degrees,array,sqrt # to allow arrays, their functions and types

from bluesky.tools.aero import kts, ft
from bluesky.core import Entity
from bluesky.stack import command
from .windfield import Windfield


class WindSim(Entity, Windfield, replaceable=True):      
    @command(name='WIND')
    def add(self, lat: 'lat', lon: 'lon', *winddata: 'float'):
        """ Define a wind vector as part of the 2D or 3D wind field.
        
            Arguments:
            - lat/lon: Horizonal position to define wind vector(s)
            - winddata: 
              - If the wind at this location is independent of altitude
                winddata has two elements:
                - direction [degrees]
                - speed (magnitude) [knots]
              - If the wind varies with altitude winddata has three elements:
                - altitude [ft]
                - direction [degrees]
                - speed (magnitude) [knots]
                In this case, repeating combinations of alt/dir/spd can be provided
                to specify wind at multiple altitudes.
        """
        ndata = len(winddata)

        # No altitude or just one: same wind for all altitudes at this position
        if ndata == 2 or (ndata == 3 and winddata[0] is None): # only one point, ignore altitude
            if winddata[-2] is None or winddata[-1] is None:
               return False, "Wind direction and speed needed."

            self.addpoint(lat,lon,winddata[-2],winddata[-1]*kts)

        # More than one altitude is given
        elif ndata >= 3:
            windarr = array(winddata)
            dirarr = windarr[1::3]
            spdarr = windarr[2::3] * kts
            altarr = windarr[0::3] * ft

            self.addpoint(lat,lon,dirarr,spdarr,altarr)
            
        elif winddata.count("DEL") > 0:
            self.clear()

        else:# Something is wrong
            return False, "Winddata not recognized"

        return True

    @command(name='GETWIND')
    def get(self, lat: 'lat', lon: 'lon', alt: 'alt'=None):
        """ Get wind at a specified position (and optionally at altitude) 
        
            Arguments:
            - lat, lon: Horizontal position where wind should be determined [deg]
            - alt: Altitude at which wind should be determined [ft]
        """
        vn,ve = self.getdata(lat,lon,alt)

        wdir = (degrees(arctan2(ve,vn)) + 180.) % 360.
        wspd = sqrt(vn * vn + ve * ve)

        txt  = "WIND AT %.5f, %.5f: %03d/%d" % (lat,lon,round(wdir),round(wspd/kts))

        return True, txt