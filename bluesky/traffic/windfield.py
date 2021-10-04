""" Wind implementation for BlueSky."""
from numpy import array, sin, cos, arange, radians, ones, append, ndarray, \
                  minimum, repeat, delete, zeros, maximum, floor, interp, \
                  pi, concatenate, unique
from scipy.interpolate import interp1d, RegularGridInterpolator
from bluesky.tools.aero import ft

class Windfield():
    """ Windfield class:
        Methods:
            clear()    = clear windfield, no wind vectors defined

            addpoint(lat,lon,winddir,winddspd,windalt=None)
                       = add a wind vector to a position,
                         windvector can be arrays for altitudes (optional)
                         returns index of vector (0,1,2,3,..)
                         all units are SI units, angles in degrees

            get(lat,lon,alt=0)
                       = get wind vector for given position and optional
                         altitude, all can be arrays,
                         vnorth and veast will be returned in same dimension

            remove(idx) = remove a defined profile using the index

        Members:
            lat(nvec)          = latitudes of wind definitions
            lon(nvec)          = longitudes of wind definitions
            altaxis(nalt)      = altitude axis (fixed, 250 m resolution)

            vnorth(nalt,nvec)  = wind north component [m/s]
            veast(nalt,nvec)   = wind east component [m/s]

            winddim   = Windfield dimension, will automatically be detected:
                          0 = no wind
                          1 = constant wind
                          2 = 2D field (no alt profiles),
                          3 = 3D field (alt dependent wind at some points)

    """
    def __init__(self):
        # For altitude use fixed axis to allow vectorisation later
        self.altmax  = 45000. * ft   # [m]
        self.altstep = 100. * ft    # [m]

        # Axis
        self.altaxis = arange(0., self.altmax + self.altstep, self.altstep)
        self.idxalt  = arange(0, len(self.altaxis), 1.)
        self.nalt    = len(self.altaxis)

        # List of indices of points with an altitude profile (for 3D check)
        self.iprof   = []

        # Clear actual field
        self.clear()
        return

    def clear(self): #Clear actual field
        # Windfield dimension will automatically be detected:
        # 0 = no wind, 1 = constant wind, 2 = 2D field (no alt profiles),
        # 3 = 3D field (alt matters), used to speed up interpolation
        self.winddim = 0
        self.lat     = array([])
        self.lon     = array([])
        self.vnorth  = array([[]])
        self.veast   = array([[]])
        self.nvec    = 0
        self.fe      = None
        self.fn      = None
        return

    def addpointvne(self, lat, lon, vnorth, veast, windalt=None):
        """ Add a vector of lat/lon positions (arrays) with a (2D vector of) 
            wind speed [m/s] in north and east component. 
            Optionally an array with altitudes can be used
        """              
        if windalt is not None and len(windalt) > 1:           
            # Set altitude interpolation functions
            fnorth = interp1d(windalt, vnorth.T, bounds_error=False, 
                              fill_value=(vnorth[0], vnorth[-1]), assume_sorted=True)
            feast  = interp1d(windalt, veast.T, bounds_error=False, 
                              fill_value=(veast[0], veast[-1]), assume_sorted=True)
                       
            # Assume regular grid and set RGI for interpolation
            if len(lat) > 3: 
                try:
                    # Interpolate along windalt axis
                    altaxis = concatenate((array([0.]), windalt))
                    vnaxis = fnorth(altaxis).T
                    veaxis = feast(altaxis).T
                    
                    # Get unique latitudes and longitudes for RGI
                    lats = unique(lat)
                    lons = unique(lon)
                    
                    # Set RGI interpolation functions
                    vevalues = veaxis.reshape((len(altaxis), len(lats), len(lons)))
                    vnvalues = vnaxis.reshape((len(altaxis), len(lats), len(lons)))
                    self.fe = RegularGridInterpolator((altaxis, lats, lons), 
                                                      vevalues, bounds_error=False, fill_value=0.)
                    self.fn = RegularGridInterpolator((altaxis, lats, lons), 
                                                      vnvalues, bounds_error=False, fill_value=0.) 
                except:
                    # Create vn, ve if RGI is not possible
                    vnaxis = fnorth(self.altaxis).T
                    veaxis = feast(self.altaxis).T
            else:
                # Create vn, ve if less than 4 coords are present
                vnaxis = fnorth(self.altaxis).T
                veaxis = feast(self.altaxis).T
        
            self.winddim = 3
            self.iprof.append(len(self.lat) + 1)
        
        else:
            vnaxis = vnorth
            veaxis = veast

        self.nvec += len(lat)
        self.lat = append(self.lat, lat)
        self.lon = append(self.lon, lon)

        if self.vnorth.size == 0:
            self.vnorth = vnaxis
            self.veast  = veaxis
        else:
            self.vnorth = concatenate((self.vnorth, vnaxis), axis=1) 
            self.veast  = concatenate((self.veast, veaxis), axis=1)

        if self.winddim<3: # No 3D => set dim to 0,1 or 2 dep on nr of points
            self.winddim = min(2,len(self.lat))

    def addpoint(self,lat,lon,winddir,windspd,windalt=None):
        """ addpoint: adds a lat,lon position with a wind direction [deg]
                                                     and wind speedd [m/s]
            Optionally an array with altitudes can be used in which case windspd
            and wind speed need to have the same dimension
        """

        # If scalar, copy into table for altitude axis
        if not(type(windalt) in [ndarray,list]) and windalt == None: # scalar to array
            prof3D = False # no wind profile, just one value
            wspd   = ones(self.nalt)*windspd
            wdir   = ones(self.nalt)*winddir
            vnaxis = wspd*cos(radians(wdir)+pi)
            veaxis = wspd*sin(radians(wdir)+pi)

        # if list or array, convert to alt axis of wind field
        else:
            prof3D = True # switch on 3D parameter as an altitude array is given
            wspd   = array(windspd)
            wdir   = array(winddir)
            altvn  = wspd*cos(radians(wdir)+pi)
            altve  = wspd*sin(radians(wdir)+pi)
            alttab = windalt

            vnaxis = interp(self.altaxis, alttab, altvn)
            veaxis = interp(self.altaxis, alttab, altve)

#        print array([vnaxis]).transpose()
        self.lat    = append(self.lat,lat)
        self.lon    = append(self.lon,lon)

        idx = len(self.lat)-1

        if self.nvec==0:
            self.vnorth = array([vnaxis]).transpose()
            self.veast  = array([veaxis]).transpose()

        else:
            self.vnorth = append(self.vnorth,array([vnaxis]).transpose(),axis=1)
            self.veast  = append(self.veast, array([veaxis]).transpose(),axis=1)

        if self.winddim<3: # No 3D => set dim to 0,1 or 2 dep on nr of points
            self.winddim = min(2,len(self.lat))

        if prof3D:
            self.winddim = 3
            self.iprof.append(idx)

        self.nvec = self.nvec+1

        return idx # return index of added point
    
    def getdata(self,userlat,userlon,useralt=0.0): # in case no altitude specified and field is 3D, use sea level wind
        eps = 1e-20 # [m2] to avoid divison by zero for using exact same points

        swvector = (type(userlat)==list or type(userlat)==ndarray)
        if swvector:
            npos = len(userlat)
        else:
            npos = 1
        # Convert user input to right shape: columns for positions
        lat = array(userlat).reshape((1,npos))
        lon = array(userlon).reshape((1,npos))

        # Make altitude into an array, with zero or float value broadcast over npos
        if type(useralt)==ndarray:
            alt = useralt
        elif type(useralt)==list:
            alt = array(useralt)
        elif type(useralt)==float:
            alt = useralt*ones(npos)
        else:
            alt = zeros(npos)

        # Check if RGI functions are present, if so use them for interpolation
        if self.fe is not None and self.fn is not None:
            vnorth = self.fn(concatenate((alt.reshape(1,-1), lat, lon), axis=0).T)
            veast  = self.fe(concatenate((alt.reshape(1,-1), lat, lon), axis=0).T)
        else:
            # Check dimension of wind field
            if self.winddim == 0:   # None = no wind
                vnorth = zeros(npos)
                veast  = zeros(npos)
    
            elif self.winddim == 1: # Constant = one point defined, so constant wind
                vnorth = ones(npos)*self.vnorth[0,0]
                veast  = ones(npos)*self.veast[0,0]
    
            elif self.winddim >= 2: # 2D/3D field = more points defined but no altitude profile
    
                #---- Get horizontal weight factors
    
                # Average cosine for flat-eartyh approximation
                cavelat = cos(radians(0.5*(lat+array([self.lat]).transpose())))
    
                # Lat and lon distance in 60 nm units (1 lat degree)
                dy = lat - array([self.lat]).transpose() #(nvec,npos)
                dx = cavelat*(lon - array([self.lon]).transpose())
    
                # Calulate invesre distance squared
                invd2   = 1./(eps+dx*dx+dy*dy) # inverse of distance squared
    
                # Normalize weights
                sumsid2 = ones((1,self.nvec)).dot(invd2) # totals to normalize weights
                totals = repeat(sumsid2,self.nvec,axis=0) # scale up dims to (nvec,npos)
                
                horfact = invd2/totals # rows x col = nvec x npos, weight factors
    
                #---- Altitude interpolation
    
                # No altitude profiles used: do 2D planar interpolation only
                if self.winddim == 2 or ((type(useralt) not in (list,ndarray)) and useralt==0.0): # 2D field no altitude interpolation
                    vnorth  = self.vnorth[0,:].dot(horfact)
                    veast   = self.veast[0,:].dot(horfact)
    
                # 3D interpolation as one or more points contain altitude profile
                else:
    
                    # Get altitude index as float for alt interpolation
                    idxalt = maximum(0., minimum(self.altaxis[-1]-eps, alt) / self.altstep) # find right index
    
                    # Convert to index and factor
                    ialt   = floor(idxalt).astype(int) # index array for lower altitude
                    falt   = idxalt-ialt  # factor for upper value
    
                    # Altitude interpolation combined with horizontal
                    nvec   = len(self.lon) # Get number of definition points
    
                    # North wind (y-direction ot lat direction)
                    vn0    = (self.vnorth[ialt,:]*horfact.T).dot(ones((nvec,1))) # hor interpolate lower alt (npos x)
                    vn1    = (self.vnorth[ialt+1,:]*horfact.T).dot(ones((nvec,1))) # hor interpolate lower alts (npos x)
                    vnorth = (1.-falt)*(vn0.reshape(npos)) + falt*(vn1.reshape(npos)) # As 1D array
    
                    # East wind (x-direction or lon direction)
                    ve0    = (self.veast[ialt,:]*horfact.T).dot(ones((nvec,1)))
                    ve1    = (self.veast[ialt+1,:]*horfact.T).dot(ones((nvec,1)))
                    veast  = (1.-falt)*(ve0.reshape(npos)) + falt*(ve1.reshape(npos)) # As 1D array

        # Return same type as positons were given
        if type(userlat)==ndarray:
            return vnorth,veast

        elif type(userlat)==list:
            return list(vnorth),list(veast)

        else:
            return float(vnorth),float(veast)

    def remove(self,idx): # remove a point using the returned index when it was added
        if idx<len(self.lat):
            self.lat = delete(self.lat,idx)
            self.lon = delete(self.lat,idx)

            self.vnorth = delete(self.vnorth,idx,axis=1)
            self.veast  = delete(self.veast ,idx,axis=1)

            if idx in self.iprof:
                self.iprof.remove(idx)

            if self.winddim<3 or len(self.iprof)==0 or len(self.lat)==0:
                self.winddim = min(2,len(self.lat)) # Check for 0, 1D, 2D or 3D

        return
