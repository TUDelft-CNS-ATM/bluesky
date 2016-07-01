import numpy as np
from ...tools.aero import ft, nm

# Import default CD methods
try:
    import cStateBasedCD as StateBasedCD
except:
    import StateBasedCD

# Import default CR methods
import DoNothing
import Eby
import MVP
import Swarm


class ASAS():
    """ Central class for ASAS conflict detection and resolution.
        Maintains a confict database, and links to external CD and CR methods."""

    # Dictionary of CD methods
    CDmethods = {"STATEBASED": StateBasedCD}

    # Dictionary of CR methods
    CRmethods = {"OFF": DoNothing, "MVP": MVP, "EBY": Eby, "SWARM": Swarm}

    @classmethod
    def addCDMethod(asas, name, module):
        asas.CDmethods[name] = module

    @classmethod
    def addCRMethod(asas, name, module):
        asas.CRmethods[name] = module

    # Constructor of conflict database, call with SI units (meters and seconds)
    def __init__(self, tlook, R, dh):
        self.t0asas      = -999.          # last time ASAS was called
        self.dtasas      = 1.00           # interval for ASAS
        self.swasas      = True           # [-] whether to perform CD&R
        self.dtlookahead = tlook          # [s] lookahead time

        self.mar         = 1.05           # [-] Safety margin for evasion
        self.R           = R              # [m] Horizontal separation minimum
        self.dh          = dh             # [m] Vertical separation minimum
        self.Rm          = R * self.mar   # [m] Horizontal separation minimum + margin
        self.dhm         = dh * self.mar  # [m] Vertical separation minimum + margin

        self.vmin        = 100.0          # [m/s] Minimum ASAS velocity
        self.vmax        = 180.0          # [m/s] Maximum ASAS velocity

        self.cd_name     = "STATEBASED"
        self.cr_name     = "OFF"
        self.cd          = ASAS.CDmethods[self.cd_name]
        self.cr          = ASAS.CRmethods[self.cr_name]

        # Reset database
        self.reset()

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
        self.confpairs    = []      # Start with emtpy database: no conflicts
        self.nconf        = 0       # Number of detected conflicts
        self.latowncpa    = np.array([])
        self.lonowncpa    = np.array([])
        self.altowncpa    = np.array([])

        self.conflist_all = []  # List of all Conflicts
        self.LOSlist_all  = []  # List of all Losses Of Separation
        self.conflist_exp = []  # List of all Conflicts in experiment time
        self.LOSlist_exp  = []  # List of all Losses Of Separation in experiment time
        self.conflist_now = []  # List of current Conflicts
        self.LOSlist_now  = []  # List of current Losses Of Separation

        # For keeping track of locations with most severe intrusions
        self.LOSmaxsev    = []
        self.LOShmaxsev   = []
        self.LOSvmaxsev   = []

        # ASAS info per aircraft:
        self.iconf        = []            # index in 'conflicting' aircraft database
        self.asasactive   = np.array([], dtype=bool)  # whether the autopilot follows ASAS or not
        self.asashdg      = np.array([])  # heading provided by the ASAS [deg]
        self.asasspd      = np.array([])  # speed provided by the ASAS (eas) [m/s]
        self.asasalt      = np.array([])  # speed alt by the ASAS [m]
        self.asasvsp      = np.array([])  # speed vspeed by the ASAS [m/s]

    def create(self, hdg, spd, alt):
        # ASAS info: no conflict => empty list
        self.iconf.append([])  # List of indices in 'conflicting' aircraft database

        # ASAS output commanded values
        self.asasactive = np.append(self.asasactive, False)
        self.asashdg    = np.append(self.asashdg, hdg)
        self.asasspd    = np.append(self.asasspd, spd)
        self.asasalt    = np.append(self.asasalt, alt)
        self.asasvsp    = np.append(self.asasvsp, 0.)

    def delete(self, idx):
        del self.iconf[idx]
        self.asasactive = np.delete(self.asasactive, idx)
        self.asashdg    = np.delete(self.asashdg, idx)
        self.asasspd    = np.delete(self.asasspd, idx)
        self.asasalt    = np.delete(self.asasalt, idx)
        self.asasvsp    = np.delete(self.asasvsp, idx)

    def update(self, traf, simt):
        # Scheduling: when dt has passed or restart:
        if self.swasas and self.t0asas + self.dtasas < simt or simt < self.t0asas:
            self.t0asas       = simt

            # Reset lists before new CD
            self.iconf        = [[] for ac in range(traf.ntraf)]
            self.nconf        = 0
            self.confpairs    = []
            self.latowncpa    = []
            self.lonowncpa    = []
            self.altowncpa    = []

            self.LOSlist_now  = []
            self.conflist_now = []

            # Conflict detection and resolution
            self.cd.detect(self, traf, simt)
            self.cr.resolve(self, traf)
