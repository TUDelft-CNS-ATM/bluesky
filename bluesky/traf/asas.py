import numpy as np
from ..tools.aero import ft, nm

# Import default CD methods
try:
    from CDRmethods import cStateBasedCD as StateBasedCD
except:
    from CDRmethods import StateBasedCD

# Import default CR methods
from CDRmethods import DoNothing, Eby, MVP, Swarm


class ASAS():
    """ Central class for ASAS conflict detection and resolution.
        Maintains a confict database, and links to external CD and CR methods."""

    # Dictionary of CD methods
    CDmethods = {"StateBased": StateBasedCD}

    # Dictionary of CR methods
    CRmethods = {"OFF": DoNothing, "MVP": MVP, "Eby": Eby, "Swarm": Swarm}

    # Constructor of conflict database, call with SI units (meters and seconds)
    def __init__(self, tlook, R, dh):
        self.t0asas = -999.  # last time ASAS was called
        self.dtasas = 1.00  # interval for ASAS

        self.swasas = True      # [-] whether to perform CD&R
        self.dtlookahead = tlook     # [s] lookahead time

        self.mar = 1.05      # [-] Safety margin for evasion
        self.R = R         # [m] Horizontal separation minimum
        self.dh = dh        # [m] Vertical separation minimum
        self.Rm = R * self.mar   # [m] Horizontal separation minimum + margin
        self.dhm = dh * self.mar  # [m] Vertical separation minimum + margin

        self.vmin = 100.0     # [m/s] Minimum ASAS velocity
        self.vmax = 180.0     # [m/s] Maximum ASAS velocity

        self.reset()                 # Reset database

        self.cd_name = "StateBased"
        self.cr_name = "OFF"

        self.cd = ASAS.CDmethods[self.cd_name]
        self.cr = ASAS.CRmethods[self.cr_name]

    def toggle(self, flag=None):
        if flag is None:
            return True, "ASAS is currently " + ("ON" if self.swasas else "OFF")
        self.swasas = flag

    def SetCDmethod(self, method=""):
        if method is "":
            return True, ("Current CD method: " + self.cd_name +
                        "\nAvailable CD methods: " + str.join(", ", ASAS.CDmethods.keys()))
        if method not in ASAS.CDmethods:
            return False, (method + " doesn't exist.\nAvailable CD methods: " + str.join(", ", ASAS.CDmethods.keys()))

        self.cd_name = method
        self.cd = ASAS.CDmethods[method]

    def SetCRmethod(self, method=""):
        if method is "":
            return True, ("Current CR method: " + self.cr_name +
                        "\nAvailable CR methods: " + str.join(", ", ASAS.CRmethods.keys()))
        if method not in ASAS.CRmethods:
            return False, (method + " doesn't exist.\nAvailable CR methods: " + str.join(", ", ASAS.CRmethods.keys()))

        self.cr_name = method
        self.cr = ASAS.CRmethods[method]
        self.cr.start(self)

    def SetPZR(self, value=None):
        if value is None:
            return True, ("ZONER [radius (nm)]\nCurrent PZ radius: %.2f NM" % self.R / nm)

        self.R  = value * nm
        self.Rm = np.maximum(self.mar * self.R, self.Rm)

    def SetPZH(self, value=None):
        if value is None:
            return True, ("ZONEDH [height (ft)]\nCurrent PZ height: %.2f ft" % self.dh / ft)

        self.dh  = value * ft
        self.dhm = np.maximum(self.mar * self.dh, self.dhm)

    def SetPZRm(self, value=None):
        if value is None:
            return True, ("RSZONER [radius (nm)]\nCurrent PZ radius margin: %.2f NM" % self.Rm / nm)

        if value < self.R / nm:
            return False, "PZ radius margin may not be smaller than PZ radius"

        self.Rm  = value * nm

    def SetPZHm(self, value=None):
        if value is None:
            return True, ("RSZONEDH [height (ft)]\nCurrent PZ height margin: %.2f ft" % self.dhm / ft)

        if value < self.dh / ft:
            return False, "PZ height margin may not be smaller than PZ height"

        self.dhm  = value * ft

    def SetDtLook(self, value=None):
        if value is None:
            return True, ("DTLOOK [time]\nCurrent value: %.1f sec" % self.dtlookahead)

        self.dtlookahead = value

    def SetDtNoLook(self, value=None):
        if value is None:
            return True, ("DTNOLOOK [time]\nCurrent value: %.1f sec" % self.dtasas)

        self.dtasas = value

    # Reset conflict database
    def reset(self):
        self.conf      = []      # Start with emtpy database: no conflicts
        self.nconf     = 0       # Number of detected conflicts
        self.swconfl   = np.array([])
        self.latowncpa = np.array([])
        self.lonowncpa = np.array([])
        self.altowncpa = np.array([])
        self.latintcpa = np.array([])
        self.lonintcpa = np.array([])
        self.altintcpa = np.array([])

        self.idown = []
        self.idoth = []

        self.conflist_all = []  # List of all Conflicts
        self.LOSlist_all = []  # List of all Losses Of Separation
        self.conflist_exp = []  # List of all Conflicts in experiment time
        self.LOSlist_exp = []  # List of all Losses Of Separation in experiment time
        self.conflist_now = []  # List of current Conflicts
        self.LOSlist_now = []  # List of current Losses Of Separation

        # For keeping track of locations with most severe intrusions
        self.LOSmaxsev = []
        self.LOShmaxsev = []
        self.LOSvmaxsev = []

        # ASAS info per aircraft:
        self.iconf      = []            # index in 'conflicting' aircraft database
        self.asasactive = np.array([])  # whether the autopilot follows ASAS or not
        self.asashdg    = np.array([])  # heading provided by the ASAS [deg]
        self.asasspd    = np.array([])  # speed provided by the ASAS (eas) [m/s]
        self.asasalt    = np.array([])  # speed alt by the ASAS [m]
        self.asasvsp    = np.array([])  # speed vspeed by the ASAS [m/s]

        self.inconflict = np.array([], dtype=bool)

    def create(self, hdg, spd, alt):
        # ASAS info: no conflict => -1
        self.iconf.append(-1)  # index in 'conflicting' aircraft database

        # ASAS output commanded values
        self.asasactive = np.append(self.asasactive, False)
        self.asashdg    = np.append(self.asashdg, hdg)
        self.asasspd    = np.append(self.asasspd, spd)
        self.asasalt    = np.append(self.asasalt, alt)
        self.asasvsp    = np.append(self.asasvsp, 0.)
        self.inconflict = np.append(self.inconflict, False)

    def delete(self, idx):
        del self.iconf[idx]
        self.asasactive = np.delete(self.asasactive, idx)
        self.asashdg    = np.delete(self.asashdg, idx)
        self.asasspd    = np.delete(self.asasspd, idx)
        self.asasalt    = np.delete(self.asasalt, idx)
        self.asasvsp    = np.delete(self.asasvsp, idx)
        self.inconflict = np.delete(self.inconflict, idx)

    def update(self, traf, simt):
        # Scheduling: when dt has passed or restart:
        if self.swasas and self.t0asas + self.dtasas < simt or simt < self.t0asas:
            self.t0asas = simt

            # Call with traffic database and sim data
            self.cd.detect(self, traf, simt)
            self.APorASAS(traf)
            self.cr.resolve(self, traf)

    #============================= Trajectory Recovery ============================

    # Decide for each aircraft whether the ASAS should be followed or not
    def APorASAS(self, traf):
        # Indicate for all A/C that they should follow their Autopilot
        self.asasactive.fill(False)
        self.inconflict.fill(False)

        # Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.conflist_all:
            id1, id2 = self.ConflictToIndices(traf, conflict)
            if id1 != "Fail":
                pastCPA = self.ConflictIsPastCPA(traf, id1, id2)

                if not pastCPA:

                    # Indicate that the A/C must follow their ASAS
                    self.asasactive[id1] = True
                    self.inconflict[id1] = True

                    self.asasactive[id2] = True
                    self.inconflict[id2] = True

    #========================= Check if past CPA ==================================
    def ConflictIsPastCPA(self, traf, id1, id2):

        d = np.array([traf.lon[id2] - traf.lon[id1], traf.lat[id2] - traf.lat[id1], traf.alt[id2] - traf.alt[id1]])

        # find track in degrees
        t1 = np.radians(traf.trk[id1])
        t2 = np.radians(traf.trk[id2])

        # write velocities as vectors and find relative velocity vector
        v1 = np.array([np.sin(t1) * traf.tas[id1], np.cos(t1) * traf.tas[id1], traf.vs[id1]])
        v2 = np.array([np.sin(t2) * traf.tas[id2], np.cos(t2) * traf.tas[id2], traf.vs[id2]])
        v = np.array(v2 - v1)

        pastCPA = np.dot(d, v) > 0

        return pastCPA

    #====================== Give A/C indices of conflict pair =====================
    def ConflictToIndices(self, traf, conflict):
        ac1, ac2 = conflict.split(" ")

        try:
            id1 = traf.id.index(ac1)
            id2 = traf.id.index(ac2)
        except:
            return "Fail", "Fail"

        return id1, id2
