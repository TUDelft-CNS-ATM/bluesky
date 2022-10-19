""" BlueSky deletion area plugin. This plugin can use an area definition to
    delete aircraft that exit the area. Statistics on these flights can be
    logged with the FLSTLOG logger. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import datalog, areafilter
from bluesky.core import Entity, timed_function
from bluesky.tools.aero import ft,kts,nm,fpm

# Log parameters for the flight statistics log
flstheader = \
    '#######################################################\n' + \
    'FLST LOG\n' + \
    'Flight Statistics\n' + \
    '#######################################################\n\n' + \
    'Parameters [Units]:\n' + \
    'Deletion Time [s], ' + \
    'Call sign [-], ' + \
    'Spawn Time [s], ' + \
    'Flight time [s], ' + \
    'Actual Distance 2D [nm], ' + \
    'Actual Distance 3D [nm], ' + \
    'Work Done [MJ], ' + \
    'Latitude [deg], ' + \
    'Longitude [deg], ' + \
    'Altitude [ft], ' + \
    'TAS [kts], ' + \
    'Vertical Speed [fpm], ' + \
    'Heading [deg], ' + \
    'Origin Lat [deg], ' + \
    'Origin Lon [deg], ' + \
    'Destination Lat [deg], ' + \
    'Destination Lon [deg], ' + \
    'ASAS Active [bool], ' + \
    'Pilot ALT [ft], ' + \
    'Pilot SPD (TAS) [kts], ' + \
    'Pilot HDG [deg], ' + \
    'Pilot VS [fpm]'  + '\n'

confheader = \
    '#######################################################\n' + \
    'CONF LOG\n' + \
    'Conflict Statistics\n' + \
    '#######################################################\n\n' + \
    'Parameters [Units]:\n' + \
    'Simulation time [s], ' + \
    'Total number of conflicts in exp area [-]\n'

# Global data
area = None

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global area
    area = Area()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'AREA',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
        }

    stackfunctions = {
        'AREA': [
            'AREA Shapename/OFF or AREA lat,lon,lat,lon,[top,bottom]',
            '[float/txt,float,float,float,alt,alt]',
            area.set_area,
            'Define deletion area (aircraft leaving area are deleted)'
        ],
        'EXP': [
            'EXP Shapename/OFF or EXP lat,lon,lat,lon,[top,bottom]',
            '[float/txt,float,float,float,alt,alt]',
            lambda *args: area.set_area(*args, exparea=True),
            'Define experiment area (area of interest)'
        ],
        'TAXI': [
            'TAXI ON/OFF [alt] : OFF auto deletes traffic below 1500 ft',
            'onoff[,alt]',
            area.set_taxi,
            'Switch on/off ground/low altitude mode, prevents auto-delete at 1500 ft'
        ]
    }
    # init_plugin() should always return these two dicts.
    return config, stackfunctions

class Area(Entity):
    ''' Traffic area: delete traffic when it leaves this area (so not when outside)'''
    def __init__(self):
        super().__init__()
        # Parameters of area
        self.active = False
        self.delarea = ''
        self.exparea = ''
        self.swtaxi = True  # Default ON: Doesn't do anything. See comments of set_taxi function below.
        self.swtaxialt = 1500.0  # Default alt for TAXI OFF
        self.prevconfpairs = set()
        self.confinside_all = 0

        # The FLST logger
        self.flst = datalog.crelog('FLSTLOG', None, flstheader)
        self.conflog = datalog.crelog('CONFLOG', None, confheader)

        with self.settrafarrays():
            self.insdel = np.array([], dtype=bool) # In deletion area or not
            self.insexp = np.array([], dtype=bool) # In experiment area or not
            self.oldalt = np.array([])
            self.distance2D = np.array([])
            self.distance3D = np.array([])
            self.dstart2D = np.array([])
            self.dstart3D = np.array([])
            self.workstart = np.array([])
            self.entrytime = np.array([])
            self.create_time = np.array([])

    def reset(self):
        ''' Reset area state when simulation is reset. '''
        super().reset()
        self.active = False
        self.delarea = ''
        self.exparea = ''
        self.swtaxi = True
        self.swtaxialt = 1500.0
        self.confinside_all = 0

    def create(self, n=1):
        ''' Create is called when new aircraft are created. '''
        super().create(n)
        self.oldalt[-n:] = traf.alt[-n:]
        self.insdel[-n:] = False
        self.insexp[-n:] = False
        self.create_time[-n:] = sim.simt

    @timed_function(name='AREA', dt=1.0)
    def update(self, dt):
        ''' Update flight efficiency metrics
            2D and 3D distance [m], and work done (force*distance) [J] '''
        if self.active:
            resultantspd = np.sqrt(traf.gs * traf.gs + traf.vs * traf.vs)
            self.distance2D += dt * traf.gs
            self.distance3D += dt * resultantspd

            # Find out which aircraft are currently inside the experiment area, and
            # determine which aircraft need to be deleted.
            insdel = areafilter.checkInside(self.delarea, traf.lat, traf.lon, traf.alt)
            insexp = insdel if not self.exparea else \
                areafilter.checkInside(self.exparea, traf.lat, traf.lon, traf.alt)
            # Find all aircraft that were inside in the previous timestep, but no
            # longer are in the current timestep
            delidx = np.where(np.array(self.insdel) * (np.array(insdel) == False))[0]
            self.insdel = insdel

            # Count new conflicts where at least one of the aircraft is inside
            # the experiment area
            # Store statistics for all new conflict pairs
            # Conflict pairs detected in the current timestep that were not yet
            # present in the previous timestep
            confpairs_new = list(set(traf.cd.confpairs) - self.prevconfpairs)
            if confpairs_new:
                # If necessary: select conflict geometry parameters for new conflicts
                # idxdict = dict((v, i) for i, v in enumerate(traf.cd.confpairs))
                # idxnew = [idxdict.get(i) for i in confpairs_new]
                # dcpa_new = np.asarray(traf.cd.dcpa)[idxnew]
                # tcpa_new = np.asarray(traf.cd.tcpa)[idxnew]
                # tLOS_new = np.asarray(traf.cd.tLOS)[idxnew]
                # qdr_new = np.asarray(traf.cd.qdr)[idxnew]
                # dist_new = np.asarray(traf.cd.dist)[idxnew]

                newconf_unique = {frozenset(pair) for pair in confpairs_new}
                ac1, ac2 = zip(*newconf_unique)
                idx1 = traf.id2idx(ac1)
                idx2 = traf.id2idx(ac2)
                newconf_inside = np.logical_or(insexp[idx1], insexp[idx2])

                nnewconf_exp = np.count_nonzero(newconf_inside)
                if nnewconf_exp:
                    self.confinside_all += nnewconf_exp
                    self.conflog.log(self.confinside_all)
            self.prevconfpairs = set(traf.cd.confpairs)

            # Register distance values upon entry of experiment area
            newentries = np.logical_not(self.insexp) * insexp
            self.dstart2D[newentries] = self.distance2D[newentries]
            self.dstart3D[newentries] = self.distance3D[newentries]
            self.workstart[newentries] = traf.work[newentries]
            self.entrytime[newentries] = sim.simt

            # Log flight statistics when exiting experiment area
            exits = np.logical_and(self.insexp,np.logical_not(insexp))
            # Update insexp
            self.insexp = insexp

            if np.any(exits):
                self.flst.log(
                    np.array(traf.id)[exits],
                    self.create_time[exits],
                    sim.simt - self.entrytime[exits],
                    (self.distance2D[exits] - self.dstart2D[exits])/nm,
                    (self.distance3D[exits] - self.dstart3D[exits])/nm,
                    (traf.work[exits] - self.workstart[exits])*1e-6,
                    traf.lat[exits],
                    traf.lon[exits],
                    traf.alt[exits]/ft,
                    traf.tas[exits]/kts,
                    traf.vs[exits]/fpm,
                    traf.hdg[exits],
                    traf.cr.active[exits],
                    traf.aporasas.alt[exits]/ft,
                    traf.aporasas.tas[exits]/kts,
                    traf.aporasas.vs[exits]/fpm,
                    traf.aporasas.hdg[exits])

            # delete all aicraft in self.delidx
            if len(delidx) > 0:
                traf.delete(delidx)



        # Autodelete for descending with swTaxi:
        if not self.swtaxi:
            delidxalt = np.where((self.oldalt >= self.swtaxialt)
                                 * (traf.alt < self.swtaxialt))[0]
            self.oldalt = traf.alt
            if len(delidxalt) > 0:
                traf.delete(list(delidxalt))

    def set_area(self, *args, exparea=False):
        ''' Set Experiment Area. Aircraft leaving the experiment area are deleted.
        Input can be existing shape name, or a box with optional altitude constraints.'''
        curname = self.exparea if exparea else self.delarea
        msgname = 'Experiment area' if exparea else 'Deletion area'
        # if all args are empty, then print out the current area status
        if not args:
            return True, f'{msgname} is currently ON (name={curname})' if self.active else \
                         f'{msgname} is currently OFF'

        # start by checking if the first argument is a string -> then it is an area name
        if isinstance(args[0], str) and len(args)==1:
            if areafilter.hasArea(args[0]):
                # switch on Area, set it to the shape name
                if exparea:
                    self.exparea = args[0]
                else:
                    self.delarea = args[0]

                self.active = True
                self.flst.start()
                self.conflog.start()
                return True, f'{msgname} is set to {args[0]}'
            if args[0][:2] =='OF':
                # switch off the area and reset the logger
                self.active = False
                return True, f'{msgname} is switched OFF'
            if args[0][:2] == 'ON':
                if not self.name:
                    return False, 'No area defined.'
                else:
                    self.active = True
                    return True, f'{msgname} switched ON (name={curname})'
            # shape name is unknown
            return False, 'Shapename unknown. ' + \
                'Please create shapename first or shapename is misspelled!'
        # if first argument is a float -> then make a box with the arguments
        if isinstance(args[0],(float, int)) and 4<=len(args)<=6:
            self.active = True
            if exparea:
                self.exparea = 'EXPAREA'
                areafilter.defineArea('EXPAREA', 'BOX', args[:4], *args[4:])
            else:
                self.delarea = 'DELAREA'
                areafilter.defineArea('DELAREA', 'BOX', args[:4], *args[4:])
            self.flst.start()
            self.conflog.start()
            return True, f'{msgname} is ON. Area name is: {"EXP" if exparea else "DEL"}AREA'

        return False,  'Incorrect arguments' + \
                       '\nAREA Shapename/OFF or\n Area lat,lon,lat,lon,[top,bottom]'

    def set_taxi(self, flag,alt=1500*ft):
        ''' Taxi ON/OFF to autodelete below a certain altitude if taxi is off'''
        self.swtaxi = flag # True =  taxi allowed, False = autodelete below swtaxialt
        self.swtaxialt = alt
