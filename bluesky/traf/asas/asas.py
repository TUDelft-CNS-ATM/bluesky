import numpy as np
from ... import settings
from ...tools.aero import ft, nm

# Import default CD methods
try:
    import casas as StateBasedCD
except ImportError:
    StateBasedCD = False

if not settings.prefer_compiled or not StateBasedCD:
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

    def __init__(self):
        # All ASAS variables are initialized in the reset function
        self.reset()

    def reset(self):
        """ ASAS constructor """
        self.cd_name      = "STATEBASED"
        self.cr_name      = "OFF"
        self.cd           = ASAS.CDmethods[self.cd_name]
        self.cr           = ASAS.CRmethods[self.cr_name]

        self.dtasas       = settings.asas_dt           # interval for ASAS
        self.dtlookahead  = settings.asas_dtlookahead  # [s] lookahead time
        self.mar          = settings.asas_mar          # [-] Safety margin for evasion
        self.R            = settings.asas_pzr * nm     # [m] Horizontal separation minimum for detection
        self.dh           = settings.asas_pzh * ft     # [m] Vertical separation minimum for detection
        self.Rm           = self.R * self.mar          # [m] Horizontal separation minimum for resolution 
        self.dhm          = self.dh * self.mar         # [m] Vertical separation minimum for resolution 
        self.swasas       = True                       # [-] whether to perform CD&R
        self.tasas        = 0.0                        # Next time ASAS should be called

        self.vmin         = 51.4                       # [m/s] Minimum ASAS velocity (100 kts)
        self.vmax         = 308.6                      # [m/s] Maximum ASAS velocity (600 kts)
        self.vsmin        = -3000./60.*ft              # [m/s] Minimum ASAS vertical speed        
        self.vsmax        = 3000./60.*ft               # [m/s] Maximum ASAS vertical speed   
        
        self.swresohoriz  = False                      # [-] switch to limit resolution to the horizontal direction
        self.swresospd    = False                      # [-] switch to use only speed resolutions (works with swresohoriz = True)
        self.swresohdg    = False                      # [-] switch to use only heading resolutions (works with swresohoriz = True)        
        self.swresovert   = False                      # [-] switch to limit resolution to the vertical direction       
        self.swresocoop   = False                      # [-] switch to limit resolution magnitude to half (cooperative resolutions) 
        
        self.swprio       = False                      # [-] switch to activate priority rules for conflict resolution
        self.priocode     = "FF1"                      # [-] Code of the priority rule that is to be used (FF1, FF2, FF3, LAY1, LAY2)
        
        self.noresolst    = []                         # [-] list for NORESO command. Nobody will avoid conflicts with aircraft in this list
        
        self.resoFacH     = 1.0                        # [-] set horizontal resolution factor (1.0 = 100%)
        self.resoFacV     = 1.0                        # [-] set horizontal resolution factor (1.0 = 100%)

        self.confpairs    = []                         # Start with emtpy database: no conflicts
        self.nconf        = 0                          # Number of detected conflicts
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

    def toggle(self, flag=None):
        if flag is None:
            return True, "ASAS is currently " + ("ON" if self.swasas else "OFF")
        self.swasas = flag
        return True

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
            return True, ("ZONER [radius (nm)]\nCurrent PZ radius: %.2f NM" % (self.R / nm))

        self.R  = value * nm
        self.Rm = np.maximum(self.mar * self.R, self.Rm)

    def SetPZH(self, value=None):
        if value is None:
            return True, ("ZONEDH [height (ft)]\nCurrent PZ height: %.2f ft" % (self.dh / ft))

        self.dh  = value * ft
        self.dhm = np.maximum(self.mar * self.dh, self.dhm)

    def SetPZRm(self, value=None):
        if value is None:
            return True, ("RSZONER [radius (nm)]\nCurrent PZ radius margin: %.2f NM" % (self.Rm / nm))

        if value < self.R / nm:
            return False, "PZ radius margin may not be smaller than PZ radius"

        self.Rm  = value * nm

    def SetPZHm(self, value=None):
        if value is None:
            return True, ("RSZONEDH [height (ft)]\nCurrent PZ height margin: %.2f ft" %( self.dhm / ft))

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
    
    def SetResoHoriz(self, value=None):   
        """ Processes the RMETHH command. Sets swresovert = False"""        
        # Acceptable arguments for this command
        options = ["BOTH","SPD","HDG","NONE","ON","OFF","OF"]        
        if value is None:
            return True, "RMETHH [ON / BOTH / OFF / NONE / SPD / HDG]" + \
                         "\nHorizontal resolution limitation is currently " + ("ON" if self.swresohoriz else "OFF") + \
                         "\nSpeed resolution limitation is currently " + ("ON" if self.swresospd else "OFF") + \
                         "\nHeading resolution limitation is currently " + ("ON" if self.swresohdg else "OFF")                       
        if str(value) not in options:
            return False, "RMETH Not Understood" + "\nRMETHH [ON / BOTH / OFF / NONE / SPD / HDG]"
        else:
            if value == "ON" or value == "BOTH":
                self.swresohoriz = True
                self.swresospd   = True
                self.swresohdg   = True
                self.swresovert  = False
            elif value == "OFF" or value == "OF" or value == "NONE":
                # Do NOT swtich off self.swresovert if value == OFF
                self.swresohoriz = False
                self.swresospd   = False
                self.swresohdg   = False                
            elif value == "SPD":
                self.swresohoriz = True
                self.swresospd   = True
                self.swresohdg   = False
                self.swresovert  = False
            elif value == "HDG":
                self.swresohoriz = True
                self.swresospd   = False
                self.swresohdg   = True
                self.swresovert  = False
    
    def SetResoVert(self, value=None):   
        """ Processes the RMETHV command. Sets swresohoriz = False."""          
        # Acceptable arguments for this command
        options = ["NONE","ON","OFF","OF","V/S"]        
        if value is None:
            return True, "RMETHV [ON / V/S / OFF / NONE]" + \
                    	"\nVertical resolution limitation is currently " + ("ON" if self.swresovert else "OFF")                                           
        if str(value) not in options:
            return False, "RMETV Not Understood" + "\nRMETHV [ON / V/S / OFF / NONE]"
        else:
            if value == "ON" or value == "V/S":
                self.swresovert  = True
                self.swresohoriz = False
                self.swresospd   = False
                self.swresohdg   = False
            elif value == "OFF" or value == "OF" or value == "NONE":
                # Do NOT swtich off self.swresohoriz if value == OFF
                self.swresovert  = False
        
    def SetResoFacH(self, value=None):
        ''' Set the horizontal resolution factor'''
        if value is None:
            return True, ("RFACH [FACTOR]\nCurrent horizontal resolution factor is: %.1f" % self.resoFacH)
        
        self.resoFacH = np.abs(value)
        self.R = self.R*self.resoFacH
        self.Rm = self.R*self.mar
        
        return True, "IMPORTANT NOTE: " + \
                     "\nCurrent horizontal resolution factor is: "+ str(self.resoFacH) + \
                     "\nCurrent PZ radius:" + str(self.R/nm) + " NM" + \
                     "\nCurrent resolution PZ radius: " + str(self.Rm/nm) + " NM\n"
        
    def SetResoFacV(self, value=None):
        ''' Set the vertical resolution factor'''
        if value is None:
            return True, ("RFACV [FACTOR]\nCurrent vertical resolution factor is: %.1f" % self.resoFacV)
        
        self.resoFacV = np.abs(value)
        self.dh = self.dh*self.resoFacV
        self.dhm = self.dh*self.mar
        
        return True, "IMPORTANT NOTE: " + \
                     "\nCurrent vertical resolution factor is: "+ str(self.resoFacV) + \
                     "\nCurrent PZ height:" + str(self.dh/ft) + " ft" + \
                     "\nCurrent resolution PZ height: " + str(self.dhm/ft) + " ft\n"
                     
    def SetPrio(self, flag=None, priocode="FF1"):
        '''Set the prio switch and the type of prio '''

        options = ["FF1","FF2","FF3","LAY1","LAY2"]
        
        if flag is None:
            return True, "PRIORULES ON/OFF [PRIOCODE]"  + \
                         "\nAvailable priority codes: " + \
                         "\n     FF1:  Free Flight Primary (No Prio) " + \
                         "\n     FF2:  Free Flight Secondary (Cruising has priority)" + \
                         "\n     FF3:  Free Flight Tertiary (Climbing/descending has priority)" + \
                         "\n     LAY1: Layers Primary (Cruising has priority + horizontal resolutions)" + \
                         "\n     LAY2: Layers Secondary (Climbing/descending has priority + horizontal resolutions)" + \
                         "\nPriority is currently " + ("ON" if self.swprio else "OFF") + \
                         "\nPriority code is currently: " + str(self.priocode)                         
        
        self.swprio = flag 
        
        if priocode not in options:
            return False, "Priority code Not Understood. Available Options: " + str(options)
        else:
            self.priocode = priocode
            

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
        # Scheduling: update when dt has passed
        if self.swasas and simt >= self.tasas:
            self.tasas += self.dtasas

            # Conflict detection and resolution
            self.cd.detect(self, traf, simt)
            self.cr.resolve(self, traf)
