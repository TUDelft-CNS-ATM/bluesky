''' Conflict resolution base class. '''
import numpy as np

import bluesky as bs
from bluesky.core import Entity
from bluesky.stack import command
from bluesky.tools.aero import nm,ft


bs.settings.set_variable_defaults(asas_marh=1.01, asas_marv=1.01)


class ConflictResolution(Entity, replaceable=True):
    ''' Base class for Conflict Resolution implementations. '''
    def __init__(self):
        super().__init__()
        # [-] switch to activate priority rules for conflict resolution
        self.swprio = False  # switch priority on/off
        self.priocode = ''  # select priority mode
        self.resopairs = set()  # Resolved conflicts that are still before CPA

        # Resolution factors:
        # set < 1 to maneuver only a fraction of the resolution
        # set > 1 to add a margin to separation values
        self.resofach = bs.settings.asas_marh
        self.resofacv = bs.settings.asas_marv

        # Switches to guarantee last reso zone commands keep valid if cd zone changes
        self.resodhrelative = True # Size of resolution zone dh, vertically, set relative to CD zone
        self.resorrelative  = True # Size of resolution zone r, vertically, set relative to CD zone

        with self.settrafarrays():
            self.resooffac = np.array([], dtype=bool)
            self.noresoac = np.array([], dtype=bool)
            # whether the autopilot follows ASAS or not
            self.active = np.array([], dtype=bool)
            self.trk = np.array([])  # heading provided by the ASAS [deg]
            self.tas = np.array([])  # speed provided by the ASAS (eas) [m/s]
            self.alt = np.array([])  # alt provided by the ASAS [m]
            self.vs = np.array([])  # vspeed provided by the ASAS [m/s]

    def reset(self):
        super().reset()
        self.swprio = False
        self.priocode = ''
        self.resopairs.clear()
        self.resofach = bs.settings.asas_marh
        self.resofacv = bs.settings.asas_marv
        self.resodhrelative = True
        self.resorrelative  = True

    # By default all channels are controlled by self.active,
    # but they can be overloaded with separate variables or functions in a
    # derived ASAS Conflict Resolution class (@property decorator takes away
    # need for brackets when calling it so it can be overloaded by a variable)
    @property
    def hdgactive(self):
        ''' Return a boolean array sized according to the number of aircraft
            with True for all elements where heading is currently controlled by
            the conflict resolution algorithm.
        '''
        return self.active

    @property
    def vsactive(self):
        ''' Return a boolean array sized according to the number of aircraft
            with True for all elements where vertical speed is currently
            controlled by the conflict resolution algorithm.
        '''
        return self.active

    @property
    def altactive(self):
        ''' Return a boolean array sized according to the number of aircraft
            with True for all elements where altitude is currently controlled by
            the conflict resolution algorithm.
        '''
        return self.active

    @property
    def tasactive(self):
        ''' Return a boolean array sized according to the number of aircraft
            with True for all elements where speed is currently controlled by
            the conflict resolution algorithm.
        '''
        return self.active

    def resolve(self, conf, ownship, intruder):
        '''
            Resolve all current conflicts.
            This function should be reimplemented in a subclass for actual
            resolution of conflicts. See for instance
            bluesky.traffic.asas.mvp.
        '''
        # If resolution is off, and detection is on, and a conflict is detected
        # then asas will be active for that airplane. Since resolution is off, it
        # should then follow the auto pilot instructions.
        return ownship.ap.trk, ownship.ap.tas, ownship.ap.vs, ownship.ap.alt

    def update(self, conf, ownship, intruder):
        ''' Perform an update step of the Conflict Resolution implementation. '''
        if ConflictResolution.selected() is not ConflictResolution:
            # Only perform CR when an actual method is selected
            if conf.confpairs:
                self.trk, self.tas, self.vs, self.alt = self.resolve(conf, ownship, intruder)
            self.resumenav(conf, ownship, intruder)

    def resumenav(self, conf, ownship, intruder):
        '''
            Decide for each aircraft in the conflict list whether the ASAS
            should be followed or not, based on if the aircraft pairs passed
            their CPA.
        '''
        # Add new conflicts to resopairs and confpairs_all and new losses to lospairs_all
        self.resopairs.update(conf.confpairs)

        # Conflict pairs to be deleted
        delpairs = set()
        changeactive = dict()

        # smallest relative angle between vectors of heading a and b
        def anglediff(a, b):
            d = a - b
            if d > 180:
                return anglediff(a, b + 360)
            elif d < -180:
                return anglediff(a + 360, b)
            else:
                return d
            

        # Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)
            # If the ownship aircraft is deleted remove its conflict from the list
            if idx1 < 0:
                delpairs.add(conflict)
                continue

            if idx2 >= 0:
                # Distance vector using flat earth approximation
                re = 6371000.
                dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                      np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                              ownship.lat[idx1])),
                                      np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

                # Relative velocity vector
                vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
                                 intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])

                # Check if conflict is past CPA
                past_cpa = np.dot(dist, vrel) > 0.0

                rpz = np.max(conf.rpz[[idx1, idx2]])
                # hor_los:
                # Aircraft should continue to resolve until there is no horizontal
                # LOS. This is particularly relevant when vertical resolutions
                # are used.
                hdist = np.linalg.norm(dist)
                hor_los = hdist < rpz

                # Bouncing conflicts:
                # If two aircraft are getting in and out of conflict continously,
                # then they it is a bouncing conflict. ASAS should stay active until
                # the bouncing stops.
                is_bouncing = \
                    abs(anglediff(ownship.trk[idx1], intruder.trk[idx2])) < 30.0 and \
                    hdist < rpz * self.resofach

            # Start recovery for ownship if intruder is deleted, or if past CPA
            # and not in horizontal LOS or a bouncing conflict
            if idx2 >= 0 and (not past_cpa or hor_los or is_bouncing):
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
            else:
                # Switch ASAS off for ownship if there are no other conflicts
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If conflict is solved, remove it from the resopairs list
                delpairs.add(conflict)

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            self.active[idx] = active
            if not active:
                # Waypoint recovery after conflict: Find the next active waypoint
                # and send the aircraft to that waypoint.
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:  # To avoid problems if there are no waypoints
                    bs.traf.ap.route[idx].direct(
                        idx, bs.traf.ap.route[idx].wpname[iwpid])

        # Remove pairs from the list that are past CPA or have deleted aircraft
        self.resopairs -= delpairs

    @command(name='PRIORULES')
    def setprio(self, flag : bool = None, priocode=''):
        ''' Define priority rules (right of way) for conflict resolution. '''
        if flag is None:
            if self.__class__ is ConflictResolution:
                return False, 'No conflict resolution enabled.'
            return False, f'Resolution algorithm {self.__class__.name} hasn\'t implemented priority.'

        self.swprio = flag
        self.priocode = priocode
        return True

    @command(name='NORESO')
    def setnoreso(self, *idx : 'acid'):
        ''' ADD or Remove aircraft that nobody will avoid.
        Multiple aircraft can be sent to this function at once. '''
        if not idx:
            return True, 'NORESO [ACID, ... ] OR NORESO [GROUPID]' + \
                         '\nCurrent list of aircraft nobody will avoid:' + \
                         ', '.join(np.array(bs.traf.id)[self.noresoac])
        idx = list(idx)
        self.noresoac[idx] = np.logical_not(self.noresoac[idx])
        return True

    @command(name='RESOOFF')
    def setresooff(self, *idx : 'acid'):
        ''' ADD or Remove aircraft that will not avoid anybody else.
            Multiple aircraft can be sent to this function at once. '''
        if not idx:
            return True, 'NORESO [ACID, ... ] OR NORESO [GROUPID]' + \
                         '\nCurrent list of aircraft will not avoid anybody:' + \
                         ', '.join(np.array(bs.traf.id)[self.resooffac])
        else:
            idx = list(idx)
            self.resooffac[idx] = np.logical_not(self.resooffac[idx])
            return True

    @command(name='RFACH', aliases=('RESOFACH', 'HRFAC', 'HRESOFAC'))
    def setresofach(self, factor : float = None):
        ''' Set resolution factor horizontal
            (to maneuver only a fraction of a resolution vector)
        '''
        if factor is None:
            return True, f'RFACH [FACTOR]\nCurrent horizontal resolution factor is: {self.resofach}'
        else:
            self.resofach = factor
            self.resorrelative = True  # Size of resolution zone r, vertically, set relative to CD zone
            return True, f'Horizontal resolution factor set to {self.resofach}'

    @command(name='RFACV', aliases=('RESOFACV',))
    def setresofacv(self, factor: float = None):
        ''' Set resolution factor vertical (to maneuver only a fraction of a resolution vector). '''
        if factor is None:
            return True, f'RFACV [FACTOR]\nCurrent vertical resolution factor is: {self.resofacv}'
        self.resofacv = factor
        # Size of resolution zone dh, vertically, set relative to CD zone
        self.resodhrelative = True
        return True, f'Vertical resolution factor set to {self.resofacv}'

    @command(name='RSZONER', aliases=('RESOZONER',))
    def setresozoner(self, zoner : float = None):
        ''' Set resolution factor horizontal, but then with absolute value
            (to maneuver only a fraction of a resolution vector)
        '''
        if not bs.traf.cd.global_rpz:
            self.resorrelative = True
            return False, 'RSZONER [radiusnm]\nCan only set resolution factor when simulation contains aircraft with different RPZ,\nUse RFACH instead.'
        if zoner is None:
            return True, f'RSZONER [radiusnm]\nCurrent horizontal resolution factor is: {self.resofach}, resulting in radius: {self.resofach*bs.traf.cd.rpz_def/nm} nm'

        self.resofach = zoner / bs.traf.cd.rpz_def * nm
        # Size of resolution zone r, vertically, no longer relative to CD zone
        self.resorrelative = False
        return True, f'Horizontal resolution factor updated to {self.resofach}, resulting in radius: {zoner} nm'

    @command(name='RSZONEDH', aliases=('RESOZONEDH',))
    def setresozonedh(self, zonedh : float = None):
        '''
        Set resolution factor vertical (to maneuver only a fraction of a resolution vector),
        but then with absolute value
        '''
        if not bs.traf.cd.global_hpz:
            self.resodhrelative = True
            return False, 'RSZONEH [zonedhft]\nCan only set resolution factor when simulation contains aircraft with different HPZ,\nUse RFACV instead.'
        if zonedh is None:
            return True, f'RSZONEDH [zonedhft]\nCurrent vertical resolution factor is: {self.resofacv}, resulting in height: {self.resofacv*bs.traf.cd.hpz_def/ft} ft'

        self.resofacv = zonedh / bs.traf.cd.hpz_def * ft
        # Size of resolution zone dh, vertically, no longer relative to CD zone
        self.resodhrelative = False
        return True, f'Vertical resolution factor updated to {self.resofacv}, resulting in height: {zonedh} ft'

    @staticmethod
    @command(name='RESO')
    def setmethod(name : 'txt' = ''):
        ''' Select a Conflict Resolution method. '''
        # Get a dict of all registered CR methods
        methods = ConflictResolution.derived()
        names = ['OFF' if n == 'CONFLICTRESOLUTION' else n for n in methods]

        if not name:
            curname = 'OFF' if ConflictResolution.selected() is ConflictResolution \
                else ConflictResolution.selected().__name__
            return True, f'Current CR method: {curname}' + \
                         f'\nAvailable CR methods: {", ".join(names)}'
        # Check if the requested method exists
        if name == 'OFF':
            ConflictResolution.select()
            return True, 'Conflict Resolution turned off.'
        method = methods.get(name, None)
        if method is None:
            return False, f'{name} doesn\'t exist.\n' + \
                          f'Available CR methods: {", ".join(names)}'

        # Select the requested method
        method.select()
        return True, f'Selected {method.__name__} as CR method.'
