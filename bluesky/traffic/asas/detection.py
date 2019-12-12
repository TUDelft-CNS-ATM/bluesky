''' This module provides the Conflict Detection base class. '''
import numpy as np

import bluesky as bs
from bluesky.tools.aero import ft, nm
from bluesky.tools.replaceable import ReplaceableSingleton
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters


bs.settings.set_variable_defaults(asas_pzr=5.0, asas_pzh=1000.0,
                                  asas_dtlookahead=300.0)


class ConflictDetection(ReplaceableSingleton, TrafficArrays):
    ''' Base class for Conflict Detection implementations. '''
    def __init__(self):
        TrafficArrays.__init__(self)
        # [m] Horizontal separation minimum for detection
        self.rpz = bs.settings.asas_pzr * nm
        # [m] Vertical separation minimum for detection
        self.hpz = bs.settings.asas_pzh * ft
        # [s] lookahead time
        self.dtlookahead = bs.settings.asas_dtlookahead
        self.dtnolook = 0.0

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
        with RegisterElementParameters(self):
            self.inconf = np.array([], dtype=bool)  # In-conflict flag
            self.tcpamax = np.array([]) # Maximum time to CPA for aircraft in conflict

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
        self.inconf[:] = False

    def reset(self):
        super().reset()
        self.clearconfdb()
        self.confpairs_all.clear()
        self.lospairs_all.clear()
        self.rpz = bs.settings.asas_pzr * nm
        self.hpz = bs.settings.asas_pzh * ft
        self.dtlookahead = bs.settings.asas_dtlookahead
        self.dtnolook = 0.0

    @classmethod
    def setmethod(cls, name=''):
        ''' Select a CD method. '''
        # Get a dict of all registered CD methods
        methods = cls.derived()
        names = ['OFF' if n == 'CONFLICTDETECTION' else n for n in methods]

        if not name:
            curname = 'OFF' if cls.selected() is ConflictDetection else cls.selected().__name__
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
        return True, f'Selected {method.__name__} as CD method.'

    def setrpz(self, value=None):
        if value is None:
            return True, ("ZONER [radius (nm)]\nCurrent PZ radius: %.2f NM" % (self.rpz / nm))
        self.rpz = value * nm

    def sethpz(self, value=None):
        if value is None:
            return True, ("ZONEDH [height (ft)]\nCurrent PZ height: %.2f ft" % (self.hpz / ft))
        self.hpz = value * ft

    def setdtlook(self, value=None):
        if value is None:
            return True, ("DTLOOK [time]\nCurrent value: %.1f sec" % self.dtlookahead)
        self.dtlookahead = value

    def setdtnolook(self, value=None):
        if value is None:
            return True, ("DTNOLOOK [time]\nCurrent value: %.1f sec" % self.dtasas)
        self.dtnolook = value

    def update(self, ownship, intruder):
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
        ''' Detect any conflicts between ownship and intruder. '''
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
