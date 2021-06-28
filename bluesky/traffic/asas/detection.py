''' This module provides the Conflict Detection base class. '''
import numpy as np

import bluesky as bs
from bluesky.tools.aero import ft, nm
from bluesky.core import Entity
from bluesky.stack import command


bs.settings.set_variable_defaults(asas_pzr=5.0, asas_pzh=1000.0,
                                  asas_dtlookahead=300.0)


class ConflictDetection(Entity, replaceable=True):
    ''' Base class for Conflict Detection implementations. '''
    def __init__(self):
        super().__init__()
        ## Default values
        # [m] Horizontal separation minimum for detection
        self.rpz_def = bs.settings.asas_pzr * nm
        self.global_rpz = True
        # [m] Vertical separation minimum for detection
        self.hpz_def = bs.settings.asas_pzh * ft
        self.global_hpz = True
        # [s] lookahead time
        self.dtlookahead_def = bs.settings.asas_dtlookahead
        self.global_dtlook = True
        self.dtnolook_def = 0.0
        self.global_dtnolook = True

        # Conflicts and LoS detected in the current timestep (used for resolving)
        self.confpairs = list()
        self.lospairs = list()
        self.qdr = np.array([])
        self.dist = np.array([])
        self.dcpa = np.array([])
        self.tcpa = np.array([])
        self.tLOS = np.array([])
        # Unique conflicts and LoS in the current timestep (a, b) = (b, a)
        self.confpairs_unique = set()
        self.lospairs_unique = set()

        # All conflicts and LoS since simt=0
        self.confpairs_all = list()
        self.lospairs_all = list()

        # Per-aircraft conflict data
        with self.settrafarrays():
            self.inconf = np.array([], dtype=bool)  # In-conflict flag
            self.tcpamax = np.array([]) # Maximum time to CPA for aircraft in conflict
            # [m] Horizontal separation minimum for detection
            self.rpz = np.array([])
            # [m] Vertical separation minimum for detection
            self.hpz = np.array([])
            # [s] lookahead time
            self.dtlookahead = np.array([])
            self.dtnolook = np.array([])

    def clearconfdb(self):
        ''' Clear conflict database. '''
        self.confpairs_unique.clear()
        self.lospairs_unique.clear()
        self.confpairs.clear()
        self.lospairs.clear()
        self.qdr = np.array([])
        self.dist = np.array([])
        self.dcpa = np.array([])
        self.tcpa = np.array([])
        self.tLOS = np.array([])
        self.inconf = np.zeros(bs.traf.ntraf)
        self.tcpamax = np.zeros(bs.traf.ntraf)

    def create(self, n):
        super().create(n)
        # Initialise values of own states
        self.rpz[-n:] = self.rpz_def
        self.hpz[-n:] = self.hpz_def
        self.dtlookahead[-n:] = self.dtlookahead_def
        self.dtnolook[-n:] = self.dtnolook_def

    def reset(self):
        super().reset()
        self.clearconfdb()
        self.confpairs_all.clear()
        self.lospairs_all.clear()
        self.rpz_def = bs.settings.asas_pzr * nm
        self.hpz_def = bs.settings.asas_pzh * ft
        self.dtlookahead_def = bs.settings.asas_dtlookahead
        self.dtnolook_def = 0.0
        self.global_rpz = self.global_hpz = True
        self.global_dtlook = self.global_dtnolook = True

    @staticmethod
    @command(name='CDMETHOD', aliases=('ASAS',))
    def setmethod(name : 'txt' = ''):
        ''' Select a Conflict Detection (CD) method. '''
        # Get a dict of all registered CD methods
        methods = ConflictDetection.derived()
        names = ['OFF' if n == 'CONFLICTDETECTION' else n for n in methods]
        if not name:
            curname = 'OFF' if ConflictDetection.selected() is ConflictDetection \
                else ConflictDetection.selected().__name__
            return True, f'Current CD method: {curname}' + \
                         f'\nAvailable CD methods: {", ".join(names)}'
        # Check if the requested method exists
        if name == 'OFF':
            # Select the base method and clear the conflict database
            ConflictDetection.select()
            ConflictDetection.instance().clearconfdb()
            return True, 'Conflict Detection turned off.'
        if name == 'ON':
            # Just select the first CD method in the list
            name = next(n for n in names if n != 'OFF')
        method = methods.get(name, None)
        if method is None:
            return False, f'{name} doesn\'t exist.\n' + \
                          f'Available CD methods: {", ".join(names)}'

        # Select the requested method
        method.select()
        ConflictDetection.instance().clearconfdb()
        return True, f'Selected {method.__name__} as CD method.'

    @command(name='ZONER', aliases=('PZR', 'RPZ', 'PZRADIUS'))
    def setrpz(self, radius: float = -1.0, *acidx: 'acid'):
        ''' Set the horizontal separation distance (i.e., the radius of the
            protected zone) in nautical miles. 
            
            Arguments:
            - radius: The protected zone radius in nautical miles
            - acidx: Aircraft id(s) or group. When this argument is not provided the default PZ radius is changed.
              Otherwise the PZ radius for the passed aircraft is changed. '''
        if radius < 0.0:
            return True, f'ZONER [radius(nm), acid(s)/ac group]\nCurrent default PZ radius: {self.rpz_def / nm:.2f} NM'
        if len(acidx) > 0:
            if isinstance(acidx[0], np.ndarray):
                acidx = acidx[0]
            self.rpz[acidx] = radius * nm
            self.global_rpz = False
            return True, f'Setting PZ radius to {radius} NM for {len(acidx)} aircraft'
        oldradius = self.rpz_def
        self.rpz_def = radius * nm
        if self.global_rpz:
            self.rpz[:] = self.rpz_def
        # Adjust factors for reso zone if those were set with an absolute value
        if not bs.traf.cr.resorrelative:
            bs.stack.stack(f"RSZONER {bs.traf.cr.resofach*oldradius/nm}")
        return True, f'Setting default PZ radius to {radius} NM'

    @command(name='ZONEDH', aliases=('PZDH', 'DHPZ', 'PZHEIGHT'))
    def sethpz(self, height: float = -1.0, *acidx: 'acid'):
        ''' Set the vertical separation distance (i.e., half of the protected
            zone height) in feet.

            Arguments:
            - height: The vertical separation height in feet
            - acidx: Aircraft id(s) or group. When this argument is not provided the default PZ height is changed.
              Otherwise the PZ height for the passed aircraft is changed. '''
        if height < 0.0:
            return True, f'ZONEDH [height (ft), acid(s)/ac group]\nCurrent default PZ height: {self.hpz / ft:.2f} ft'
        if len(acidx) > 0:
            if isinstance(acidx[0], np.ndarray):
                acidx = acidx[0]
            self.hpz[acidx] = height * ft
            self.global_hpz = False
            return True, f'Setting PZ height to {height} ft for {len(acidx)} aircraft'
        oldhpz = self.hpz_def
        self.hpz_def = height * ft
        if self.global_hpz:
            self.hpz[:] = self.hpz_def
        # Adjust factors for reso zone if those were set with an absolute value
        if not bs.traf.cr.resodhrelative:
            bs.stack.stack(f"RSZONEDH {bs.traf.cr.resofacv*oldhpz/ft}")
        return True, f'Setting default PZ height to {height} ft'

    @command(name='DTLOOK')
    def setdtlook(self, time: 'time' = -1.0, *acidx: 'acid'):
        ''' Set the lookahead time (in [hh:mm:]sec) for conflict detection. '''
        if time < 0.0:
            return True, f'DTLOOK[time]\nCurrent value: {self.dtlookahead_def: .1f} sec'
        if len(acidx) > 0:
            if isinstance(acidx[0], np.ndarray):
                acidx = acidx[0]
            self.dtlookahead[acidx] = time
            self.global_dtlook = False
            return True, f'Setting CD lookahead to {time} sec for {len(acidx)} aircraft'
        self.dtlookahead_def = time
        if self.global_dtlook:
            self.dtlookahead[:] = time
        return True, f'Setting default CD lookahead to {time} sec'

    @command(name='DTNOLOOK')
    def setdtnolook(self, time: 'time' = -1.0, *acidx: 'acid'):
        ''' Set the interval (in [hh:mm:]sec) in which conflict detection
            is skipped after a conflict resolution. '''
        if time < 0.0:
            return True, f'DTNOLOOK[time]\nCurrent value: {self.dtnolook_def: .1f} sec'
        if len(acidx) > 0:
            if isinstance(acidx[0], np.ndarray):
                acidx = acidx[0]
            self.dtnolook[acidx] = time
            self.global_dtnolook = False
            return True, f'Setting CD no-look to {time} sec for {len(acidx)} aircraft'
        self.dtnolook_def = time
        if self.global_dtnolook:
            self.dtnolook[:] = time
        return True, f'Setting default CD no-look to {time} sec'

    def update(self, ownship, intruder):
        ''' Perform an update step of the Conflict Detection implementation. '''
        self.confpairs, self.lospairs, self.inconf, self.tcpamax, self.qdr, \
            self.dist, self.dcpa, self.tcpa, self.tLOS = \
                self.detect(ownship, intruder, self.rpz, self.hpz, self.dtlookahead)

        # confpairs has conflicts observed from both sides (a, b) and (b, a)
        # confpairs_unique keeps only one of these
        confpairs_unique = {frozenset(pair) for pair in self.confpairs}
        lospairs_unique = {frozenset(pair) for pair in self.lospairs}

        self.confpairs_all.extend(confpairs_unique - self.confpairs_unique)
        self.lospairs_all.extend(lospairs_unique - self.lospairs_unique)

        # Update confpairs_unique and lospairs_unique
        self.confpairs_unique = confpairs_unique
        self.lospairs_unique = lospairs_unique

    def detect(self, ownship, intruder, rpz, hpz, dtlookahead):
        ''' Detect any conflicts between ownship and intruder.
            This function should be reimplemented in a subclass for actual
            detection of conflicts. See for instance
            bluesky.traffic.asas.statebased.
        '''
        confpairs = []
        lospairs = []
        inconf = np.zeros(ownship.ntraf)
        tcpamax = np.zeros(ownship.ntraf)
        qdr = np.array([])
        dist = np.array([])
        dcpa = np.array([])
        tcpa = np.array([])
        tLOS = np.array([])
        return confpairs, lospairs, inconf, tcpamax, qdr, dist, dcpa, tcpa, tLOS
