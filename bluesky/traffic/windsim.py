''' Simulate wind in BlueSky. '''
from numpy import arctan2,degrees,array,sqrt # to allow arrays, their functions and types

from bluesky.tools.aero import kts
from bluesky.core import Entity
from bluesky.stack import command
from .windfield import Windfield


class WindSim(Entity, Windfield, replaceable=True):
    @command(name='WIND')
    def add(self, lat: 'lat', lon: 'lon', *winddata: 'alt/float'):
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
            altarr = windarr[0::3]

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

    def add_HR(self):
        print("Hello")
        #print(WindSim.interpolation([1,1,1,1,1,1], [1,1,1,7,9,21], [1,1,5,5,0,4], [1,1,5,3,20,6], [1,11,7,5,13,18], [1,11,7,3,1,1], [1,11,-1,5,0,2], [1,11,-1,3,2,5], [3,1,1,1,1,1], \
        #                            [3,1,1,7,9,21], [3,1,5,5,0,4], [3,1,5,3,20,6], [3,11,7,5,2,4], [3,11,7,3,0,0], [3,11,-1,5,10,16], [3,11,-1,3,8,8], 2,6,3,4))

    #[time,lat,lon,height,uwind,vwind]
    def interpolation(time_lat_lon_h_mmmm , time_lat_lon_h_mmmp, time_lat_lon_h_mmpm , time_lat_lon_h_mmpp , time_lat_lon_h_mpmm , time_lat_lon_h_mpmp , time_lat_lon_h_mppm , \
                     time_lat_lon_h_mppp , time_lat_lon_h_pmmm, time_lat_lon_h_pmmp , time_lat_lon_h_pmpm , time_lat_lon_h_pmpp , time_lat_lon_h_ppmm , time_lat_lon_h_ppmp , \
                     time_lat_lon_h_pppm , time_lat_lon_h_pppp, time , lat , lon, height):

        time_lat_lon_h_mmm = [time_lat_lon_h_mmmm[0] , time_lat_lon_h_mmmm[1] , time_lat_lon_h_mmmm[2], height, WindSim.interpolate(height, time_lat_lon_h_mmmm[3], time_lat_lon_h_mmmp[3], time_lat_lon_h_mmmm[4] , time_lat_lon_h_mmmp[4]) , \
                              WindSim.interpolate(height, time_lat_lon_h_mmmm[3], time_lat_lon_h_mmmp[3], time_lat_lon_h_mmmm[5] , time_lat_lon_h_mmmp[5])]
        time_lat_lon_h_mmp = [time_lat_lon_h_mmpm[0], time_lat_lon_h_mmpm[1], time_lat_lon_h_mmpm[2], height, WindSim.interpolate(height, time_lat_lon_h_mmpm[3], time_lat_lon_h_mmpp[3], time_lat_lon_h_mmpm[4], time_lat_lon_h_mmpp[4]), \
                              WindSim.interpolate(height, time_lat_lon_h_mmpm[3], time_lat_lon_h_mmpp[3], time_lat_lon_h_mmpm[5], time_lat_lon_h_mmpp[5])]
        time_lat_lon_h_mpm = [time_lat_lon_h_mpmm[0], time_lat_lon_h_mpmm[1], time_lat_lon_h_mpmm[2], height, WindSim.interpolate(height, time_lat_lon_h_mpmm[3], time_lat_lon_h_mpmp[3], time_lat_lon_h_mpmm[4], time_lat_lon_h_mpmp[4]), \
                              WindSim.interpolate(height, time_lat_lon_h_mpmm[3], time_lat_lon_h_mpmp[3], time_lat_lon_h_mpmm[5], time_lat_lon_h_mpmp[5])]
        time_lat_lon_h_mpp = [time_lat_lon_h_mppm[0], time_lat_lon_h_mppm[1], time_lat_lon_h_mppm[2], height, WindSim.interpolate(height, time_lat_lon_h_mppm[3], time_lat_lon_h_mppp[3], time_lat_lon_h_mppm[4], time_lat_lon_h_mppp[4]), \
                              WindSim.interpolate(height, time_lat_lon_h_mppm[3], time_lat_lon_h_mppp[3], time_lat_lon_h_mppm[5], time_lat_lon_h_mppp[5])]
        time_lat_lon_h_pmm = [time_lat_lon_h_pmmm[0], time_lat_lon_h_pmmm[1], time_lat_lon_h_pmmm[2], height, WindSim.interpolate(height, time_lat_lon_h_pmmm[3], time_lat_lon_h_pmmp[3], time_lat_lon_h_pmmm[4], time_lat_lon_h_pmmp[4]), \
                              WindSim.interpolate(height, time_lat_lon_h_pmmm[3], time_lat_lon_h_pmmp[3], time_lat_lon_h_pmmm[5], time_lat_lon_h_pmmp[5])]
        time_lat_lon_h_pmp = [time_lat_lon_h_pmpm[0], time_lat_lon_h_pmpm[1], time_lat_lon_h_pmpm[2], height, WindSim.interpolate(height, time_lat_lon_h_pmpm[3], time_lat_lon_h_pmpp[3], time_lat_lon_h_pmpm[4], time_lat_lon_h_pmpp[4]), \
                              WindSim.interpolate(height, time_lat_lon_h_pmpm[3], time_lat_lon_h_pmpp[3], time_lat_lon_h_pmpm[5], time_lat_lon_h_pmpp[5])]
        time_lat_lon_h_ppm = [time_lat_lon_h_ppmm[0], time_lat_lon_h_ppmm[1], time_lat_lon_h_ppmm[2], height, WindSim.interpolate(height, time_lat_lon_h_ppmm[3], time_lat_lon_h_ppmp[3], time_lat_lon_h_ppmm[4], time_lat_lon_h_ppmp[4]), \
                              WindSim.interpolate(height, time_lat_lon_h_ppmm[3], time_lat_lon_h_ppmp[3], time_lat_lon_h_ppmm[5], time_lat_lon_h_ppmp[5])]
        time_lat_lon_h_ppp = [time_lat_lon_h_pppm[0], time_lat_lon_h_pppm[1], time_lat_lon_h_pppm[2], height, WindSim.interpolate(height, time_lat_lon_h_pppm[3], time_lat_lon_h_pppp[3], time_lat_lon_h_pppm[4], time_lat_lon_h_pppp[4]), \
                              WindSim.interpolate(height, time_lat_lon_h_pppm[3], time_lat_lon_h_pppp[3], time_lat_lon_h_pppm[5], time_lat_lon_h_pppp[5])]

        time_lat_lon_h_mm  = [time_lat_lon_h_mmm[0], time_lat_lon_h_mmm[1], lon, height, WindSim.interpolate(lon, time_lat_lon_h_mmm[2], time_lat_lon_h_mmp[2], time_lat_lon_h_mmm[4], time_lat_lon_h_mmp[4]), \
                              WindSim.interpolate(lon, time_lat_lon_h_mmm[2], time_lat_lon_h_mmp[2], time_lat_lon_h_mmm[5], time_lat_lon_h_mmp[5])]
        time_lat_lon_h_mp = [time_lat_lon_h_mpm[0], time_lat_lon_h_mpm[1], lon, height, WindSim.interpolate(lon, time_lat_lon_h_mpm[2], time_lat_lon_h_mpp[2], time_lat_lon_h_mpm[4], time_lat_lon_h_mpp[4]), \
                             WindSim.interpolate(lon, time_lat_lon_h_mpm[2], time_lat_lon_h_mpp[2], time_lat_lon_h_mpm[5], time_lat_lon_h_mpp[5])]
        time_lat_lon_h_pm = [time_lat_lon_h_pmm[0], time_lat_lon_h_pmm[1], lon, height, WindSim.interpolate(lon, time_lat_lon_h_pmm[2], time_lat_lon_h_pmp[2], time_lat_lon_h_pmm[4], time_lat_lon_h_pmp[4]), \
                             WindSim.interpolate(lon, time_lat_lon_h_pmm[2], time_lat_lon_h_pmp[2], time_lat_lon_h_pmm[5], time_lat_lon_h_pmp[5])]
        time_lat_lon_h_pp = [time_lat_lon_h_ppm[0], time_lat_lon_h_ppm[1], lon, height, WindSim.interpolate(lon, time_lat_lon_h_ppm[2], time_lat_lon_h_ppp[2], time_lat_lon_h_ppm[4], time_lat_lon_h_ppp[4]), \
                             WindSim.interpolate(lon, time_lat_lon_h_ppm[2], time_lat_lon_h_ppp[2], time_lat_lon_h_ppm[5], time_lat_lon_h_ppp[5])]

        time_lat_lon_h_m  = [time_lat_lon_h_mm[0], lat, lon, height, WindSim.interpolate(lat, time_lat_lon_h_mm[1], time_lat_lon_h_mp[1], time_lat_lon_h_mm[4], time_lat_lon_h_mp[4]), \
                             WindSim.interpolate(lat, time_lat_lon_h_mm[1], time_lat_lon_h_mp[1], time_lat_lon_h_mm[5], time_lat_lon_h_mp[5])]
        time_lat_lon_h_p  = [time_lat_lon_h_pm[0], lat, lon, height, WindSim.interpolate(lat, time_lat_lon_h_pm[1], time_lat_lon_h_pp[1], time_lat_lon_h_pm[4],  time_lat_lon_h_pp[4]), \
                            WindSim.interpolate(lat, time_lat_lon_h_pm[1], time_lat_lon_h_pp[1], time_lat_lon_h_pm[5], time_lat_lon_h_pp[5])]

        time_lat_lon_h    = [time, lat, lon, height, WindSim.interpolate(time, time_lat_lon_h_m[0], time_lat_lon_h_p[0], time_lat_lon_h_m[4],  time_lat_lon_h_p[4]), \
                            WindSim.interpolate(time, time_lat_lon_h_m[0], time_lat_lon_h_p[0], time_lat_lon_h_m[5], time_lat_lon_h_p[5])]

#           [1,0,0,5,5,4] , [1,0,8,5,6,3] --> [1, 0, 4, 5, 5.5, 3.5]
#           [1,4,0,5,7,6] , [1,4,8,5,6,3] --> [1, 4, 4, 5, 6.5, 4.5]
#                                                                       [1, 2, 4, 5, 6, 4]
#           [7,0,0,5,5,4] , [7,0,8,5,8,5] --> [7, 0, 4, 5, 6.5, 4.5]
#           [7,4,0,5,7,6] , [7,4,8,5,6,3] --> [7, 4, 4, 5, 6.5, 4.5]                                        [4,2,4,5,6.25,4.25]
#                                                                       [7, 2, 4, 5, 6.5, 4.5]
#         print(time_lat_lon_h_m)
#         print(time_lat_lon_h_p)
        return time_lat_lon_h


        print("interpol")

    def interpolate(value , value_a , value_b, answer_a , answer_b):
        if value_a != value_b:
            answer = (value - value_a)/(value_b - value_a) * (answer_b-answer_a) + answer_a
            return answer
        return answer_a
