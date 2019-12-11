""" Airborne Separation Assurance System. Implements CD&R functionality together with
    separate conflict detection and conflict resolution modules."""
import numpy as np
import bluesky as bs
from bluesky import settings
from bluesky.tools.simtime import timed_function
from bluesky.tools.aero import ft, nm
from bluesky.tools.trafficarrays import TrafficArrays, RegisterElementParameters
from .detection import ConflictDetection
from .resolution import ConflictResolution

# Register settings defaults
settings.set_variable_defaults(asas_dt=1.0, asas_vmin=200.0, asas_vmax=500.0)


class ASAS(TrafficArrays):
    """ Central class for ASAS conflict detection and resolution.
        Maintains a confict database, and links to external CD and CR methods."""
    def __init__(self):
        TrafficArrays.__init__(self)
        self.cd = ConflictDetection()
        self.cr = ConflictResolution()

        with RegisterElementParameters(self):
            # ASAS info PER AIRCRAFT:
            self.tcpamax = np.array([])  # Maximum time to CPA for aircraft in conflict
            self.trk = np.array([])  # heading provided by the ASAS [deg]
            self.tas = np.array([])  # speed provided by the ASAS (eas) [m/s]
            self.alt = np.array([])  # alt provided by the ASAS [m]
            self.vs = np.array([])  # vspeed provided by the ASAS [m/s]
            # ASAS-visualization on SSD
            self.asasn = np.array([])  # [m/s] North resolution speed from ASAS
            self.asase = np.array([])  # [m/s] East resolution speed from ASAS

        # All ASAS variables are initialized in the reset function
        self.reset()

    def reset(self):
        super(ASAS, self).reset()
        self.swasas       = True                            # [-] whether to perform CD&R

        # Sets of pairs: conflict pairs, LoS pairs
        self.confpairs_unique = set()  # Unique conflict pairs (a, b) = (b, a) are merged
        self.lospairs_unique = set()  # Unique LOS pairs (a, b) = (b, a) are merged
        self.confpairs_all = list()  # All conflicts since simt=0
        self.lospairs_all = list()  # All losses of separation since simt=0

    def toggle(self, flag=None):
        if flag is None:
            return True, "ASAS is currently " + ("ON" if self.swasas else "OFF")
        self.swasas = flag

        # Clear conflict list when switched off
        if not self.swasas:
            self.clearconfdb()
            self.inconf = self.inconf&False  # Set in-conflict flag to False

        return True

    def clearconfdb(self):
        """
        Clear conflict database
        """
        self.confpairs_unique = set()  # Unique conflict pairs (a, b) = (b, a) are merged
        self.lospairs_unique = set()  # Unique LOS pairs (a, b) = (b, a) are merged
        self.confpairs_all = list()  # All conflicts since simt=0
        self.lospairs_all = list()  # All losses of separation since simt=0

    def create(self, n=1):
        super(ASAS, self).create(n)

        self.trk[-n:] = bs.traf.trk[-n:]
        self.tas[-n:] = bs.traf.tas[-n:]
        self.alt[-n:] = bs.traf.alt[-n:]
        self.vs[-n:] = bs.traf.vs[-n:]

    @timed_function('asas', dt=settings.asas_dt)
    def update(self):
        if not self.swasas or bs.traf.ntraf == 0:
            return

        # Conflict detection
        self.cd.update(bs.traf, bs.traf)

        # Conflict resolution if there are conflicts
        if self.cd.confpairs:
            self.trk, self.tas, self.vs, self.alt = self.cr.update(self.cd, bs.traf, bs.traf)
            # Stores resolution vector

        self.asase = np.where(self.cd.inconf, self.tas * np.sin(self.trk / 180 * np.pi), 0.0)
        self.asasn = np.where(self.cd.inconf, self.tas * np.cos(self.trk / 180 * np.pi), 0.0)

        # confpairs has conflicts observed from both sides (a, b) and (b, a)
        # confpairs_unique keeps only one of these
        confpairs_unique = {frozenset(pair) for pair in self.cd.confpairs}
        lospairs_unique = {frozenset(pair) for pair in self.cd.lospairs}

        self.confpairs_all.extend(confpairs_unique - self.confpairs_unique)
        self.lospairs_all.extend(lospairs_unique - self.lospairs_unique)

        # Update confpairs_unique and lospairs_unique
        self.confpairs_unique = confpairs_unique
        self.lospairs_unique = lospairs_unique
