import numpy as np
from ... import settings
from ...tools.aero import ft, nm
from ...tools.dynamicarrays import DynamicArrays, RegisterElementParameters


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


class ASAS(DynamicArrays):
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

    def __init__(self, traf):
        self.traf = traf
        with RegisterElementParameters(self):
            # ASAS info per aircraft:
            self.iconf    = []            # index in 'conflicting' aircraft database

            self.active   = np.array([], dtype=bool)  # whether the autopilot follows ASAS or not
            self.trk      = np.array([])  # heading provided by the ASAS [deg]
            self.spd      = np.array([])  # speed provided by the ASAS (eas) [m/s]
            self.alt      = np.array([])  # speed alt by the ASAS [m]
            self.vs       = np.array([])  # speed vspeed by the ASAS [m/s]

        # All ASAS variables are initialized in the reset function
        self.reset()

    def reset(self):
        super(ASAS, self).reset()

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
        
        self.swnoreso     = False                      # [-] switch to activate the NORESO command. Nobody will avoid conflicts with  NORESO aircraft 
        self.noresolst    = []                         # [-] list for NORESO command. Nobody will avoid conflicts with aircraft in this list
        
        self.swresooff    = False                      # [-] switch to active the RESOOFF command. RESOOFF aircraft will NOT avoid other aircraft. Opposite of NORESO command. 
        self.resoofflst   = []                         # [-] list for the RESOOFF command. These aircraft will not do conflict resolutions. 
        
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
            return True, "PRIORULES [ON/OFF] [PRIOCODE]"  + \
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
            
    def SetNoreso(self,noresoac=''):
        '''ADD or Remove aircraft that nobody will avoid. 
        Multiple aircraft can be sent to this function at once '''
        if noresoac is '':
            return True, "NORESO [ACID]" + \
                          "\nCurrent list of aircraft nobody will avoid:" + \
                           str(self.noresolst)            
        # Split the input into separate aircraft ids if multiple acids are given
        acids = noresoac.split(',') if len(noresoac.split(',')) > 1 else noresoac.split(' ')
               
        # Remove acids if they are already in self.noresolst. This is used to 
        # delete aircraft from this list.
        # Else, add them to self.noresolst. Nobody will avoid these aircraft
        if set(acids) <= set(self.noresolst):
            self.noresolst = filter(lambda x: x not in set(acids), self.noresolst)
        else: 
            self.noresolst.extend(acids)
        
        # active the switch, if there are acids in the list
        self.swnoreso = len(self.noresolst)>0   
        
    def SetResooff(self,resooffac=''):
        "ADD or Remove aircraft that will not avoid anybody else"
        if resooffac is '':
            return True, "NORESO [ACID]" + \
                          "\nCurrent list of aircraft will not avoid anybody:" + \
                           str(self.resoofflst)            
        # Split the input into separate aircraft ids if multiple acids are given
        acids = resooffac.split(',') if len(resooffac.split(',')) > 1 else resooffac.split(' ')
               
        # Remove acids if they are already in self.resoofflst. This is used to 
        # delete aircraft from this list.
        # Else, add them to self.resoofflst. These aircraft will not avoid anybody
        if set(acids) <= set(self.resoofflst):
            self.resoofflst = filter(lambda x: x not in set(acids), self.resoofflst)
        else: 
            self.resoofflst.extend(acids)
        
        # active the switch, if there are acids in the list
        self.swresooff = len(self.resoofflst)>0  

    def create(self):
        super(ASAS, self).create()

        # ASAS output commanded values
        self.trk[-1] = self.traf.trk[-1]
        self.spd[-1] = self.traf.tas[-1]
        self.alt[-1] = self.traf.alt[-1]

    def update(self, simt):
        iconf0 = np.array(self.iconf)

        # Scheduling: update when dt has passed
        if self.swasas and simt >= self.tasas:
            self.tasas += self.dtasas

            # Conflict detection and resolution
            self.cd.detect(self, self.traf, simt)
            self.cr.resolve(self, self.traf)

        # Change labels in interface
        if settings.gui == "pygame":
            for i in range(self.traf.ntraf):
                if np.any(iconf0[i] != self.iconf[i]):
                    self.traf.label[i] = [" ", " ", " ", " "]