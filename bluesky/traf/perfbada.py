from ..settings import perf_path_bada
from glob import glob
if len(glob(perf_path_bada + '/*.OPF')) == 0:
    raise ImportError('BADA performance model: No BADA files found in ' + perf_path_bada + '!')

import os
import numpy as np
from math import *
from ..tools.aero import kts, ft, g0, a0, T0, gamma1, gamma2,  beta, R

from performance import esf, phases, limits
from ..settings import data_path


class Coefficients:
    """
    Coefficient class definition : get aircraft-specific coefficients from database

    Methods:
        coeff() : reading BADA files and store coefficients for all aircraft types

    Created by  : Lisanne Adriaens & Isabel Metz

    This class is based on:
    EUROCONTROL. User Manual for the Base of Aircraft Data (BADA) Revision 3.12, 
    EEC Technical/Scientific Report No. 14/04/24-44 edition, 2014.

    supported aircraft types (licensed users only):
    A124, A140, A148, A306, A30B, A310, A318, A319, A320, A321, A332, A333, A342, 
    A343, A345, A346, A388, A3ST, AN24, AN28, AN30, AN32, AN38, AT43, AT45, AT72, 
    AT73, AT75, ATP, B190, B350, B462, B463, B703, B712, B722, B732, B733, B734, 
    B735, B736, B737, B738, B739, B742, B743, B744, B748, B752, B753, B762, B763, 
    B764, B772, B773, B77L, B77W, B788, BA11, BE20, BE30, BE40, BE58, BE99, BE9L, 
    C130, C160, C172, C182, C25A, C25B, C25C, C421, C510, C525, C550, C551, C560, 
    C56X, C650, C680, C750, CL60, CRJ1, CRJ2, CRJ9, D228, D328, DA42, DC10, DC87, 
    DC93, DC94, DH8A, DH8C, DH8D, E120, E135, E145, E170, E190, E50P, E55P, EA50, 
    F100, F27, F28, F2TH, F50, F70, F900, FA10, FA20, FA50, FA7X, FGTH, FGTL, FGTN, 
    GL5T, GLEX, GLF5, H25A, H25B, IL76, IL86, IL96, JS32, JS41, L101, LJ35, LJ45, 
    LJ60, MD11, MD82, MD83, MU2, P180, P28A, P28U, P46T, PA27, PA31, PA34, PA44, 
    PA46, PAY2, PAY3, PC12, PRM1, RJ1H, RJ85, SB20, SF34, SH36, SR22, SU95, SW4, 
    T134, T154, T204, TB20, TB21, TBM7, TBM8, YK40, YK42
    """
    def __init__(self):
        # Check opffiles in folder 
        self.path = data_path + "/coefficients/BADA/"
        self.files = os.listdir(self.path)
        return
        
    def coeff(self):
        # create empty database
        # structure according to BADA OPF files
        self.atype = [] # aircraft type
        self.etype = [] # engine type
        self.engines = [] # engine

        # mass information
        self.mref =  [] # reference mass [t]
        self.mmin =  [] # min mass [t]
        self.mmax =  [] # max mass [t]
        self.mpyld = [] # max payload [t]
        self.gw =    [] # weight gradient on max. alt [ft/kg]

        # flight enveloppe
        self.vmo =  [] # max operating speed [kCAS]
        self.mmo =  [] # max operating mach number 
        self.hmo =  [] # max operating alt [ft]
        self.hmax = [] # max alt at MTOW and ISA [ft]
        self.gt =   [] # temp gradient on max. alt [ft/kg]

        # Surface Area [m^2]
        self.Sref = []

        # Buffet Coefficients
        self.clbo = [] # buffet onset lift coefficient
        self.k =    [] # buffet coefficient
        self.cm16 = [] # CM16
                
        # stall speeds
        self.vsto = [] # stall speed take off [knots] (CAS)
        self.vsic = [] # stall speed initial climb [knots] (CAS)
        self.vscr = [] # stall speed cruise [knots] (CAS)
        self.vsapp = [] # stall speed approach [knots] (CAS)
        self.vsld = [] # stall speed landing [knots] (CAS)

        # minimum speeds
        self.vmto = [] # minimum speed take off [knots] (CAS)
        self.vmic = [] # minimum speed initial climb [knots] (CAS)
        self.vmcr = [] # minimum speed cruise [knots] (CAS)
        self.vmap = [] # minimum speed approach [knots] (CAS)
        self.vmld = [] # minimum speed landing [knots] (CAS)
        self.cvmin = 0 # minimum speed coefficient[-]
        self.cvminto = 0 # minimum speed coefficient [-]
        
        # standard ma speeds
        self.macl = []
        self.macr = []
        self.mades = []
        
        # standard CAS speeds
        self.cascl = []
        self.cascr = []
        self.casdes = []

        # parasitic drag coefficients per phase
        self.cd0to = [] # phase takeoff
        self.cd0ic = [] # phase initial climb
        self.cd0cr = [] # phase cruise
        self.cd0ap = [] # phase approach
        self.cd0ld = [] # phase land
        self.gear = []  # drag due to gear down
        
        # induced drag coefficients per phase
        self.cd2to = [] # phase takeoff
        self.cd2ic = [] # phase initial climb
        self.cd2cr = [] # phase cruise
        self.cd2ap = [] # phase approach
        self.cd2ld = [] # phase land

        self.credj = [] # jet
        self.credt = [] # turbo

        # max climb thrust coefficients
        self.ctcth1 = [] # jet/piston [N], turboprop [ktN]
        self.ctcth2 = [] # [ft]
        self.ctcth3 = [] # jet [1/ft^2], turboprop [N], piston [ktN]

        # 1st and 2nd thrust temp coefficient 
        self.ctct1 = [] # [k]
        self.ctct2 = [] # [1/k]

        # Descent Fuel Flow Coefficients
        # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
        self.ctdesl =  [] # low alt descent thrust coefficient [-]
        self.ctdesh =  [] # high alt descent thrust coefficient [-]
        self.ctdesa =  [] # approach thrust coefficient [-]
        self.ctdesld = [] # landing thrust coefficient [-]

        # transition altitude for calculation of descent thrust
        self.hpdes = [] # [ft]

        # reference speed during descent
        self.vdes = [] # [kCAS]
        self.mdes = [] # [-]

        # Thrust specific fuel consumption coefficients
        self.cf1 = [] # jet [kg/(min*kN)], turboprop [kg/(min*kN*knot)], piston [kg/min]
        self.cf2 = [] # [knots]
        self.cf3 = [] # [kg/min]
        self.cf4 = [] # [ft]
        self.cf5 = [] # [-]

        # ground
        self.tol =  [] # take-off length[m]
        self. ldl = []#landing length[m]
        self.ws =   [] # wingspan [m]
        self.len =  [] # aircraft length[m]

        #  global parameters. 
        # Source: EUROCONTROL (2014). User Manual for the Base of Aircraft Data (BADA) Revision 3.12

        # bank angles
        # self.bank = np.deg2rad(np.array([15,35,35,35,15]))
        
        # minimum speed coefficients
        self.cvmin = 1.3
        self.cvminto = 1.2
        
        # reduced power coefficients
        self.credt = 0.25
        self.credj = 0.15
        self.credp = 0.0
                            
        # read OPF-File
        for f in self.files:
            if ".OPF" in f:
                OPFfile = open(self.path + "/" + f,'r')
                # Read-in of OPFfiles
                OPFin = OPFfile.read()
                # information is given in colums
                OPFin = OPFin.split(' ')

                # get the aircraft type
                OPFname = f[:-6]
                OPFname = OPFname.strip('_')

                # get engine type
                if OPFin[665] == "Jet":
                    etype = 1
                elif OPFin[665] == "Turboprop":
                    etype = 2
                elif OPFin[665] == "Piston":
                    etype = 3
                else:
                    etype = 1
               
                # for some aircraft, the engine name is split up in multiple elements. 
                # Here: Remove non-needed elements to avoid errors when allocating
                # the values in the BADA files to the according parameters
                engine = []
                for i in xrange (len(OPFin)):
                    if OPFname =="C160" or OPFname == "FGTH":
                        continue
                    else:
                        if OPFin[i] =="with":
                            engine.append(OPFin[i+1])
                            i = i+2
                            if OPFin[i] == "engines" or OPFin[i]=="engineswake" or OPFin[i]== "eng." or OPFin[i]=="Engines":
                                break
                            else:
                                while i<len(OPFin):
                                    if OPFin[i]!="engines" and OPFin[i]!="engineswake" and OPFin[i]!= "eng." and OPFin[i]!="Engines":
                                        OPFin.pop(i)
                                    else:
                                        break
                                break
                # only consider numeric values in the remaining document
                for i in range(len(OPFin))[::-1]:
                                   
                    if "E" not in OPFin[i]:
                        OPFin.pop(i) 
                        
                for j in range(len(OPFin))[::-1]:
                    OPFline = OPFin[j]
                    if len(OPFin[j])==0 or OPFline.isalpha() == 1:
                        OPFin.pop(j)

                    # convert all values to floats
                    while j in range(len(OPFin)):

                        try:
                            OPFin[j] = float(OPFin[j])
                            break

                        except ValueError:
                            OPFin.pop(j)

                #add the engine type
                OPFout = OPFin.append(etype)

                # format the result                  
                OPFout = np.asarray(OPFin)
                OPFout = OPFout.astype(float)

                # fill the database                
                self.atype.append(OPFname)
                self.etype.append(OPFout[70])
                self.engines.append(engine)

                # mass information
                self.mref.append(OPFout[0])
                self.mmin.append(OPFout[1])
                self.mmax.append(OPFout[2])
                self.mpyld.append(OPFout[3])
                self.gw.append(OPFout[4])

                # flight enveloppe
                self.vmo.append(OPFout[5])
                self.mmo.append(OPFout[6]) 
                self.hmo.append(OPFout[7])
                self.hmax.append(OPFout[8])
                self.gt.append(OPFout[9])

                # Surface Area [m^2]
                self.Sref.append(OPFout[10])

                # Buffet Coefficients
                self.clbo.append(OPFout[11])
                self.k.append(OPFout[12])
                self.cm16.append(OPFout[13])

                # stall speeds per phase
                self.vsto.append(OPFout[22])
                self.vsic.append(OPFout[18])
                self.vscr.append(OPFout[14])
                self.vsapp.append(OPFout[26])
                self.vsld.append(OPFout[30])

                # minimum speeds
                self.vmto.append(OPFout[22]*self.cvminto)
                self.vmic.append(OPFout[18]*self.cvmin)
                self.vmcr.append(OPFout[14]*self.cvmin)
                self.vmap.append(OPFout[26]*self.cvmin)
                self.vmld.append(OPFout[30]*self.cvmin)

                # parasitic drag coefficients per phase
                self.cd0to.append(OPFout[23])
                self.cd0ic.append(OPFout[19])
                self.cd0cr.append(OPFout[15])
                self.cd0ap.append(OPFout[27])
                self.cd0ld.append(OPFout[31])
                self.gear.append(OPFout[36])
                
                # induced drag coefficients per phase
                self.cd2to.append(OPFout[24])
                self.cd2ic.append(OPFout[20])
                self.cd2cr.append(OPFout[16])
                self.cd2ap.append(OPFout[28])
                self.cd2ld.append(OPFout[32])

                # max climb thrust coefficients
                self.ctcth1.append(OPFout[41]) # jet/piston [N], turboprop [ktN]
                self.ctcth2.append(OPFout[42]) # [ft]
                self.ctcth3.append(OPFout[43]) # jet [1/ft^2], turboprop [N], piston [ktN]

                # 1st and 2nd thrust temp coefficient 
                self.ctct1.append(OPFout[44]) # [k]
                self.ctct2.append(OPFout[45]) # [1/k]

                # Descent Fuel Flow Coefficients
                # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
                self.ctdesl.append(OPFout[46])
                self.ctdesh.append(OPFout[47])
                self.ctdesa.append(OPFout[49])
                self.ctdesld.append(OPFout[50])

                # transition altitude for calculation of descent thrust
                self.hpdes.append(OPFout[48])

                # reference speed during descent
                self.vdes.append(OPFout[51])
                self.mdes.append(OPFout[52])

                # Thrust specific fuel consumption coefficients
                self.cf1.append(OPFout[56])
                self.cf2.append(OPFout[57])
                self.cf3.append(OPFout[58])
                self.cf4.append(OPFout[59])
                self.cf5.append(OPFout[60])

                # ground
                self.tol.append(OPFout[65])
                self. ldl.append(OPFout[66])
                self.ws.append(OPFout[67])
                self.len.append(OPFout[68])         

                OPFfile.close()   

            # Airline Procedure Files
            elif ".APF" in f:
                APFfile = open(self.path+"/" + f,'r')          
            
                for line in APFfile.readlines():
                    if not line.startswith("CC") and not line.strip() =="":
                        # whitespaces for splitting the columns
                        content = line.split()

                        # for the moment only values for "average weight" (=mref),
                        # as mach values for all weight classes (low, average, high)
                        # are equal
                        if line.find("AV") != -1:
                            if float(content[5]) < 100:                        
                               self.cascl.append(float(content[4]))
                               self.macl.append(float(content[5]))
                               self.cascr.append(float(content[7]))
                               self.macr.append(float(content[8]))
                               self.mades.append(float(content[9]))  
                               self.casdes.append(float(content[10]))
                            else:
                                self.cascl.append(float(content[5]))
                                self.macl.append(float(content[6]))
                                self.cascr.append(float(content[8]))
                                self.macr.append(float(content[9]))
                                self.mades.append(float(content[10]))
                                self.casdes.append(float(content[11]))
                APFfile.close()      

        self.macl = np.array(self.macl)/100
        self.macr = np.array(self.macr)/100
        self.mades = np.array(self.mades)/100
        return


coeff = Coefficients()


class PerfBADA():

    """ 
    Performance class definition    : Aircraft performance based exclusively on open sources
    Methods:

        reset ()          : clear current database 
        create(actype)    : initialize new aircraft with performance parameters
        delete(idx)       : remove performance parameters from deleted aircraft 
        perf()            : calculate aircraft performance
        limits()          : calculate flight envelope

    Created by  : Isabel Metz
    Note: This class is based on 
        EUROCONTROL. User Manual for the Base of Aircraft Data (BADA) Revision 3.12, 
        EEC Technical/Scientific Report No. 14/04/24-44 edition, 2014.
    """
    def __init__(self, traf):
        self.traf = traf        # assign needed data from CTraffic

        self.warned = False     # Flag: Did we warn for default perf parameters yet?
        self.warned2 = False    # Flag: Use of piston engine aircraft?

        # create empty database
        self.reset()

        # prepare for coefficient readin
        coeff.coeff()

        # Flight performance scheduling
        self.dt  = 0.1           # [s] update interval of performance limits
        self.t0  = -self.dt  # [s] last time checked (in terms of simt)
        self.warned2 = False        # Flag: Did we warn for default engine parameters yet?

        return

    def engchange(self, acid, engid=None):
        return False, "BADA performance model doesn't allow changing engine type"

    def reset(self):
        """RESET DATABASE"""

        # engine
        self.etype      = np.array ([]) # jet, turboprop or piston

        # masses and dimensions
        self.mass       = np.array([]) # effective mass [kg]
        # self.mref = np.array([]) # ref. mass [kg]: 70% between min and max. mass
        self.mmin       = np.array([]) # OEW (or assumption) [kg]        
        self.mmax       = np.array([]) # MTOW (or assumption) [kg]
        # self.mpyld = np.array([]) # MZFW-OEW (or assumption) [kg]
        self.gw         = np.array([]) # weight gradient on max. alt [m/kg]
        self.Sref       = np.array([]) # wing reference surface area [m^2]
    
        # flight enveloppe
        self.vmto       = np.array([]) # min TO spd [m/s]
        self.vmic       = np.array([]) # min climb spd [m/s]
        self.vmcr       = np.array([]) # min cruise spd [m/s]
        self.vmap       = np.array([]) # min approach spd [m/s]
        self.vmld       = np.array([]) # min landing spd [m/s]   
        self.vmin       = np.array([]) # min speed over all phases [m/s]   
    
        self.vmo        =  np.array([]) # max operating speed [m/s]
        self.mmo        =  np.array([]) # max operating mach number [-]
        self.hmax       = np.array([]) # max. alt above standard MSL (ISA) at MTOW [m]
        self.hmaxact    = np.array([]) # max. alt depending on temperature gradient [m]
        self.hmo        =  np.array([]) # max. operating alt abov standard MSL [m]
        self.gt         =   np.array([]) # temp. gradient on max. alt [ft/k]
        self.maxthr     = np.array([]) # maximum thrust [N]
    
        # Buffet Coefficients
        self.clbo       = np.array([]) # buffet onset lift coefficient [-]
        self.k          = np.array([]) # buffet coefficient [-]
        self.cm16       = np.array([]) # CM16
        
        # reference CAS speeds
        self.cascl      = np.array([]) # climb [m/s]
        self.cascr      = np.array([]) # cruise [m/s]
        self.casdes     = np.array([]) # descent [m/s]
        
        #reference mach numbers [-] 
        self.macl       = np.array([]) # climb 
        self.macr       = np.array([]) # cruise 
        self.mades      = np.array([]) # descent 
        
        # parasitic drag coefficients per phase [-]
        self.cd0to      = np.array([]) # phase takeoff 
        self.cd0ic      = np.array([]) # phase initial climb
        self.cd0cr      = np.array([]) # phase cruise
        self.cd0ap      = np.array([]) # phase approach
        self.cd0ld      = np.array([]) # phase land
        self.gear       = np.array([]) # drag due to gear down
        
        # induced drag coefficients per phase [-]
        self.cd2to      = np.array([]) # phase takeoff
        self.cd2ic      = np.array([]) # phase initial climb
        self.cd2cr      = np.array([]) # phase cruise
        self.cd2ap      = np.array([]) # phase approach
        self.cd2ld      = np.array([]) # phase land
    
        # max climb thrust coefficients
        self.ctcth1      = np.array([]) # jet/piston [N], turboprop [ktN]
        self.ctcth2      = np.array([]) # [ft]
        self.ctcth3      = np.array([]) # jet [1/ft^2], turboprop [N], piston [ktN]
    
        # reduced climb power coefficient
        self.cred       = np.array([]) # [-]
        
        # 1st and 2nd thrust temp coefficient 
        self.ctct1      = np.array([]) # [k]
        self.ctct2      = np.array([]) # [1/k]
        self.dtemp      = np.array([]) # [k]
    
        # Descent Fuel Flow Coefficients
        # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
        self.ctdesl      =  np.array([]) # low alt descent thrust coefficient [-]
        self.ctdesh      =  np.array([]) # high alt descent thrust coefficient [-]
        self.ctdesa      =  np.array([]) # approach thrust coefficient [-]
        self.ctdesld     = np.array([]) # landing thrust coefficient [-]
    
        # transition altitude for calculation of descent thrust
        self.hpdes       = np.array([]) # [m]
        
        # Energy Share Factor
        self.ESF         = np.array([]) # [-]  
        
        # reference speed during descent
        self.vdes        = np.array([]) # [m/s]
        self.mdes        = np.array([]) # [-]
        
        # flight phase
        self.phase       = np.array([])
        self.post_flight = np.array([]) # taxi prior of post flight?
        self.pf_flag     = np.array([])
        
        # Thrust specific fuel consumption coefficients
        self.cf1         = np.array([]) # jet [kg/(min*kN)], turboprop [kg/(min*kN*knot)], piston [kg/min]
        self.cf2         = np.array([]) # [knots]
        self.cf3         = np.array([]) # [kg/min]
        self.cf4         = np.array([]) # [ft]
        self.cf5         = np.array([]) # [-]

        # performance
        self.Thr         = np.array([]) # thrust
        self.D           = np.array([]) # drag
        self.ff          = np.array([]) # fuel flow
    
        # ground
        self.tol         = np.array([]) # take-off length[m]
        self.ldl         = np.array([]) #landing length[m]
        self.ws          = np.array([]) # wingspan [m]
        self.len         = np.array([]) # aircraft length[m] 
        self.gr_acc      = np.array([]) # ground acceleration [m/s^2]

        return
       


    def create(self):
        actype = self.traf.type[-1]
        """CREATE NEW AIRCRAFT"""
        # note: coefficients are initialized in SI units

        # general        
        # designate aircraft to its aircraft type
        if actype in coeff.atype:
            self.coeffidx  = coeff.atype.index(actype)
        else:
            self.coeffidx  = 0
            if not self.warned:
                  print "Aircraft is using default B747-400 performance."
            self.warned    = True
        # designate aicraft to its aircraft type
        self.etype         = np.append(self.etype, coeff.etype[self.coeffidx])        
       
        # Initial aircraft mass is currently reference mass. 
        # BADA 3.12 also supports masses between 1.2*mmin and mmax
        # self.mref = np.append(self.mref, coeff.mref[self.coeffidx]*1000)         
        self.mass          = np.append(self.mass, coeff.mref[self.coeffidx]*1000)   
        self.mmin          = np.append(self.mmin, coeff.mmin[self.coeffidx]*1000)
        self.mmax          = np.append(self.mmax, coeff.mmax[self.coeffidx]*1000)
        
        # self.mpyld = np.append(self.mpyld, coeff.mpyld[self.coeffidx]*1000)
        self.gw           = np.append(self.gw, coeff.gw[self.coeffidx]*ft)
        
        # Surface Area [m^2]
        self.Sref         = np.append(self.Sref, coeff.Sref[self.coeffidx])

        # flight enveloppe
        # minimum speeds per phase
        self.vmto         = np.append(self.vmto, coeff.vmto[self.coeffidx]*kts)
        self.vmic         = np.append(self.vmic, coeff.vmic[self.coeffidx]*kts)
        self.vmcr         = np.append(self.vmcr, coeff.vmcr[self.coeffidx]*kts)
        self.vmap         = np.append(self.vmap, coeff.vmap[self.coeffidx]*kts)
        self.vmld         = np.append(self.vmld, coeff.vmld[self.coeffidx]*kts)    
        self.vmin         = np.append(self.vmin, 0.)
        self.vmo          = np.append(self.vmo, coeff.vmo[self.coeffidx]*kts)
        self.mmo          = np.append(self.mmo, coeff.mmo[self.coeffidx])
        
        # max. altitude parameters
        self.hmo          = np.append(self.hmo, coeff.hmo[self.coeffidx]*ft)        
        self.hmax         = np.append(self.hmax, coeff.hmax[self.coeffidx]*ft)
        self.hmaxact      = np.append(self.hmaxact, coeff.hmax[self.coeffidx]*ft) # initialize with hmax
        self.gt           = np.append(self.gt, coeff.gt[self.coeffidx]*ft)
        
        # max thrust setting
        self.maxthr       = np.append(self.maxthr, 1000000.) # initialize with excessive setting to avoid unrealistic limit setting

        # Buffet Coefficients
        self.clbo         = np.append(self.clbo, coeff.clbo[self.coeffidx])
        self.k            = np.append(self.k, coeff.k[self.coeffidx])
        self.cm16         = np.append(self.cm16, coeff.cm16[self.coeffidx])

        # reference speeds
        # reference CAS speeds
        self.cascl        = np.append(self.cascl, coeff.cascl[self.coeffidx]*kts)
        self.cascr        = np.append(self.cascr, coeff.cascr[self.coeffidx]*kts)
        self.casdes       = np.append(self.casdes, coeff.casdes[self.coeffidx]*kts)

        # reference mach numbers
        self.macl         = np.append(self.macl, coeff.macl[self.coeffidx])
        self.macr         = np.append(self.macr, coeff.macr[self.coeffidx] )
        self.mades        = np.append(self.mades, coeff.mades[self.coeffidx] )      

        # reference speed during descent
        self.vdes         = np.append(self.vdes, coeff.vdes[self.coeffidx]*kts)
        self.mdes         = np.append(self.mdes, coeff.mdes[self.coeffidx])
        
#######################################        


        # crossover altitude for climbing aircraft (BADA User Manual 3.12, p. 12)
        self.atranscl     = (1000/6.5)*(T0*(1-((((1+gamma1*(self.cascl/a0)**(self.cascl/a0))** \
                                (gamma2))-1) /(((1+gamma1*self.macl*self.macl)**(gamma2))-1))** \
                                    ((-(beta)*R)/g0)))        

        # crossover altitude for descending aircraft (BADA User Manual 3.12, p. 12)
        self.atransdes    = (1000/6.5)*(T0*(1-((((1+gamma1*(self.casdes/a0)*(self.casdes/a0))**(gamma2))-1) /  \
                              (((1+gamma1*self.mades*self.mades)**(gamma2))-1)) **\
                                  ((-(beta)*R)/g0)))          

       

        # aerodynamics                
        # parasitic drag coefficients per phase
        self.cd0to        = np.append(self.cd0to, coeff.cd0to[self.coeffidx])
        self.cd0ic        = np.append(self.cd0ic, coeff.cd0ic[self.coeffidx])
        self.cd0cr        = np.append(self.cd0cr, coeff.cd0cr[self.coeffidx])
        self.cd0ap        = np.append(self.cd0ap, coeff.cd0ap[self.coeffidx])
        self.cd0ld        = np.append(self.cd0ld, coeff.cd0ld[self.coeffidx])
        self.gear         = np.append(self.gear, coeff.gear[self.coeffidx])

        # induced drag coefficients per phase
        self.cd2to        = np.append(self.cd2to, coeff.cd2to[self.coeffidx])
        self.cd2ic        = np.append(self.cd2ic, coeff.cd2ic[self.coeffidx])
        self.cd2cr        = np.append(self.cd2cr, coeff.cd2cr[self.coeffidx])
        self.cd2ap        = np.append(self.cd2ap, coeff.cd2ap[self.coeffidx])
        self.cd2ld        = np.append(self.cd2ld, coeff.cd2ld[self.coeffidx])

        # reduced climb coefficient
        #jet
        if self.etype [self.traf.ntraf-1] == 1:
            self.cred     = np.append(self.cred, coeff.credj)
        # turboprop
        elif self.etype [self.traf.ntraf-1]  ==2:
            self.cred     = np.append(self.cred, coeff.credt)
        #piston
        else:
            self.cred     = np.append(self.cred, coeff.credp)

        # NOTE: model only validated for jet and turbo aircraft
        if not self.warned2 and self.etype[self.traf.ntraf-1] == 3:
            print "Using piston aircraft performance.",
            print "Not valid for real performance calculations."
            self.warned2 = True        

        # performance

        # max climb thrust coefficients
        self.ctcth1       = np.append(self.ctcth1, coeff.ctcth1[self.coeffidx]) # jet/piston [N], turboprop [ktN]
        self.ctcth2       = np.append(self.ctcth2, coeff.ctcth2[self.coeffidx]) # [ft]
        self.ctcth3       = np.append(self.ctcth3, coeff.ctcth3[self.coeffidx]) # jet [1/ft^2], turboprop [N], piston [ktN]

        # 1st and 2nd thrust temp coefficient 
        self.ctct1        = np.append(self.ctct1, coeff.ctct1[self.coeffidx]) # [k]
        self.ctct2        = np.append(self.ctct2, coeff.ctct2[self.coeffidx]) # [1/k]
        self.dtemp        = np.append(self.dtemp, 0.) # [k], difference from current to ISA temperature. At the moment: 0, as ISA environment

        # Descent Fuel Flow Coefficients
        # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
        self.ctdesl       = np.append(self.ctdesl, coeff.ctdesl[self.coeffidx])
        self.ctdesh       = np.append(self.ctdesh, coeff.ctdesh[self.coeffidx])
        self.ctdesa       = np.append(self.ctdesa, coeff.ctdesa[self.coeffidx])
        self.ctdesld      = np.append(self.ctdesld, coeff.ctdesld[self.coeffidx])

        # transition altitude for calculation of descent thrust
        self.hpdes       = np.append(self.hpdes, coeff.hpdes[self.coeffidx]*ft)
        self.ESF         = np.append(self.ESF, 1.) # neutral initialisation

        # flight phase
        self.phase       = np.append(self.phase, 0)
        self.post_flight = np.append(self.post_flight, False) # we assume prior
        self.pf_flag     = np.append(self.pf_flag, True)

         # Thrust specific fuel consumption coefficients
        self.cf1         = np.append(self.cf1, coeff.cf1[self.coeffidx])   
        
        # prevent from division per zero in fuelflow calculation
        if coeff.cf2[self.coeffidx]==0:
            self.cf2     = np.append(self.cf2, 1) 
        else:
            self.cf2     = np.append(self.cf2, coeff.cf2[self.coeffidx])
        self.cf3         = np.append(self.cf3, coeff.cf3[self.coeffidx])  
        
        # prevent from division per zero in fuelflow calculation
        if coeff.cf2[self.coeffidx]==0:
            self.cf4     = np.append(self.cf4, 1)
        else:
            self.cf4     = np.append(self.cf4, coeff.cf4[self.coeffidx])            
        self.cf5         = np.append(self.cf5, coeff.cf5[self.coeffidx])

        self.Thr         = np.append(self.Thr, 0.)
        self.D           = np.append(self.D, 0.)         
        self.ff          = np.append(self.ff, 0.)

        # ground
        self.tol         = np.append(self.tol, coeff.tol[self.coeffidx])
        self.ldl         = np.append(self.ldl, coeff.ldl[self.coeffidx])
        self.ws          = np.append(self.ws, coeff.ws[self.coeffidx])
        self.len         = np.append(self.len, coeff.len[self.coeffidx])  
        self.gr_acc      = np.append(self.gr_acc, 2.0) # value from BADA.gpf file
        # for now, BADA aircraft have the same acceleration as deceleration 
        return


    def delete(self, idx):
        """Delete REMOVED AIRCRAFT"""        
        
        # mass and dimensions
        # self.mref = np.delete(self.mref, idx)
        self.mass         = np.delete(self.mass, idx)
        self.mmin         = np.delete(self.mmin, idx)
        self.mmax         = np.delete(self.mmax, idx)
        # self.mpyld = np.delete(self.mpyld, idx)
        self.gw           = np.delete(self.gw, idx)

        self.Sref         = np.delete(self.Sref, idx)
 
        # engine
        self.etype        = np.delete(self.etype, idx)

        # flight enveloppe
        # speeds
        self.vmto         = np.delete(self.vmto, idx)
        self.vmic         = np.delete(self.vmic, idx)
        self.vmcr         = np.delete(self.vmcr, idx)
        self.vmap         = np.delete(self.vmap, idx)
        self.vmld         = np.delete(self.vmld, idx) 
        self.vmin         = np.delete(self.vmin, idx)
        self.vmo          = np.delete(self.vmo, idx)
        self.mmo          = np.delete(self.mmo, idx)

        # altitude
        self.hmo          = np.delete(self.hmo, idx)
        self.hmax         = np.delete(self.hmax, idx)
        self.hmaxact      = np.delete (self.hmaxact, idx)
        self.maxthr       = np.delete(self.maxthr, idx)
        self.gt           = np.delete(self.gt, idx)

        # buffet coefficients
        self.clbo         = np.delete(self.clbo, idx)
        self.k            = np.delete(self.k, idx)
        self.cm16         = np.delete(self.cm16, idx)

        # reference speeds        
        # reference CAS
        self.cascl        = np.delete(self.cascl, idx)
        self.cascr        = np.delete(self.cascr, idx)
        self.casdes       = np.delete(self.casdes, idx)

        # reference Mach
        self.macl         = np.delete(self.macl, idx)
        self.macr         = np.delete(self.macr, idx)
        self.mades        = np.delete(self.mades, idx)      

        # reference speed during descent
        self.vdes         = np.delete(self.vdes, idx)
        self.mdes         = np.delete(self.mdes, idx)

        # aerodynamics
        self.qS           = np.delete(self.qS, idx)

        # parasitic drag coefficients per phase
        self.cd0to        = np.delete(self.cd0to, idx)
        self.cd0ic        = np.delete(self.cd0ic, idx)
        self.cd0cr        = np.delete(self.cd0cr, idx)
        self.cd0ap        = np.delete(self.cd0ap, idx)
        self.cd0ld        = np.delete(self.cd0ld, idx)
        self.gear         = np.delete(self.gear, idx)

        # induced drag coefficients per phase
        self.cd2to        = np.delete(self.cd2to, idx)
        self.cd2ic        = np.delete(self.cd2ic, idx)
        self.cd2cr        = np.delete(self.cd2cr, idx)
        self.cd2ap        = np.delete(self.cd2ap, idx)
        self.cd2ld        = np.delete(self.cd2ld, idx)

        # performance        
        # reduced climb coefficient
        self.cred         = np.delete(self.cred, idx)

        # max climb thrust coefficients
        self.ctcth1       = np.delete(self.ctcth1, idx) # jet/piston [N], turboprop [ktN]
        self.ctcth2       = np.delete(self.ctcth2, idx) # [ft]
        self.ctcth3       = np.delete(self.ctcth3, idx) # jet [1/ft^2], turboprop [N], piston [ktN]

        # 1st and 2nd thrust temp coefficient 
        self.ctct1        = np.delete(self.ctct1, idx) # [k]
        self.ctct2        = np.delete(self.ctct2, idx) # [1/k]
        self.dtemp        = np.delete(self.dtemp, idx) # [k]

        # Descent Fuel Flow Coefficients
        self.ctdesl       = np.delete(self.ctdesl, idx)
        self.ctdesh       = np.delete(self.ctdesh, idx)
        self.ctdesa       = np.delete(self.ctdesa, idx)
        self.ctdesld      = np.delete(self.ctdesld, idx)

        # transition altitude for calculation of descent thrust
        self.hpdes       = np.delete(self.hpdes, idx)
        self.hact        = np.delete(self.hact, idx)

        # crossover altitude
        self.ESF         = np.delete (self.ESF, idx)
        self.atranscl    = np.delete(self.atranscl, idx)
        self.atransdes   = np.delete(self.atransdes, idx)

        # flight phase
        self.phase       = np.delete(self.phase, idx)
        self.bank        = np.delete(self.bank, idx)
        self.post_flight = np.delete(self.post_flight, idx)
        self.pf_flag     = np.delete(self.pf_flag, idx)

        # Thrust specific fuel consumption coefficients
        self.cf1         = np.delete(self.cf1, idx)
        self.cf2         = np.delete(self.cf2, idx)
        self.cf3         = np.delete(self.cf3, idx)
        self.cf4         = np.delete(self.cf4, idx)
        self.cf5         = np.delete(self.cf5, idx)

        self.Thr         = np.delete(self.Thr, idx)
        self.D           = np.delete(self.D, idx)         
        self.ff          = np.delete(self.ff, idx)

        # ground
        self.tol         = np.delete(self.tol, idx)
        self.ldl         = np.delete(self.ldl, idx)
        self.ws          = np.delete(self.ws, idx)
        self.len         = np.delete(self.len, idx)
        self.gr_acc      = np.delete(self.gr_acc, idx)
        
        return


    def perf(self,simt):
        if abs(simt - self.t0) >= self.dt:
            self.t0 = simt
        else:
            return
        """AIRCRAFT PERFORMANCE"""
        # BADA version
        swbada = True
        # flight phase
        self.phase, self.bank = \
        phases(self.traf.alt, self.traf.gs, self.traf.delalt, \
        self.traf.cas, self.vmto, self.vmic, self.vmap, self.vmcr, self.vmld, self.traf.bank, self.traf.bphase, \
        self.traf.hdgsel, swbada)

        # AERODYNAMICS
        # Lift
        self.qS = 0.5*self.traf.rho*np.maximum(1.,self.traf.tas)*np.maximum(1.,self.traf.tas)*self.Sref
        cl = self.mass*g0/(self.qS*np.cos(self.bank))*(self.phase!=6)+ 0.*(self.phase==6)

        # Drag
        # Drag Coefficient

        # phases TO, IC, CR
        cdph = self.cd0cr+self.cd2cr*(cl*cl)

        # phase AP
        # in case approach coefficients in OPF-Files are set to zero: 
        #Use cruise values instead
        cdapp = np.where(self.cd0ap !=0, self.cd0ap+self.cd2ap*(cl*cl), cdph)

        # phase LD
        # in case landing coefficients in OPF-Files are set to zero: 
        #Use cruise values instead
        cdld = np.where(self.cd0ld !=0, self.cd0ld+self.cd2ld*(cl*cl), cdph)        


        # now combine phases            
        cd = (self.phase==1)*cdph + (self.phase==2)*cdph + (self.phase==3)*cdph \
            + (self.phase==4)*cdapp + (self.phase ==5)*cdld  

        # Drag:
        self.D = cd*self.qS 

        # energy share factor and crossover altitude  

        # conditions
        epsalt = np.array([0.001]*self.traf.ntraf)   
        self.climb = np.array(self.traf.delalt > epsalt)
        self.descent = np.array(self.traf.delalt<-epsalt)
        lvl = np.array(np.abs(self.traf.delalt)<0.0001)*1
        


        # crossover altitiude
        atrans = self.atranscl*self.climb + self.atransdes*(1-self.climb)
        self.traf.abco = np.array(self.traf.alt>atrans)
        self.traf.belco = np.array(self.traf.alt<atrans)

        # energy share factor
        self.ESF = esf(self.traf.abco, self.traf.belco, self.traf.alt, self.traf.M,\
                  self.climb, self.descent, self.traf.delspd)

        # THRUST  
        # 1. climb: max.climb thrust in ISA conditions (p. 32, BADA User Manual 3.12)
        # condition: delta altitude positive
        self.jet = np.array(self.etype == 1)*1
        self.turbo = np.array(self.etype == 2) *1 
        self.piston = np.array(self.etype == 3)*1

    
        # temperature correction for non-ISA (as soon as applied)
        #            ThrISA = (1-self.ctct2*(self.dtemp-self.ctct1))
        # jet
        # condition
        cljet = np.logical_and.reduce([self.climb, self.jet]) *1          

        # thrust
        Tj = self.ctcth1* (1-(self.traf.alt/ft)/self.ctcth2+self.ctcth3*(self.traf.alt/ft)*(self.traf.alt/ft)) 

        # combine jet and default aircraft
        Tjc = cljet*Tj # *ThrISA
        
        # turboprop
        # condition
        clturbo = np.logical_and.reduce([self.climb, self.turbo])*1

        # thrust
        Tt = self.ctcth1/np.maximum(1.,self.traf.tas/kts)*(1-(self.traf.alt/ft)/self.ctcth2)+self.ctcth3

        # merge
        Ttc = clturbo*Tt # *ThrISA

        # piston
        clpiston = np.logical_and.reduce([self.climb, self.piston])*1            
        Tp = self.ctcth1*(1-(self.traf.alt/ft)/self.ctcth2)+self.ctcth3/np.maximum(1.,self.traf.tas/kts)
        Tpc = clpiston*Tp

        # max climb thrust for futher calculations (equals maximum avaliable thrust)
        maxthr = Tj*self.jet + Tt*self.turbo + Tp*self.piston         

        # 2. level flight: Thr = D. 
        Tlvl = lvl*self.D             

        # 3. Descent: condition: vs negative/ H>hdes: fixed formula. H<hdes: phase cr, ap, ld

        # above or below Hpdes? Careful! If non-ISA: ALT must be replaced by Hp!
        delh = (self.traf.alt - self.hpdes)
        
        # above Hpdes:  
        high = np.array(delh>0)            
        Tdesh = maxthr*self.ctdesh*np.logical_and.reduce([self.descent, high])            
               
        # below Hpdes
        low = np.array(delh<0)  
        # phase cruise
        Tdeslc = maxthr*self.ctdesl*np.logical_and.reduce([self.descent, low, (self.phase==3)])
        # phase approach
        Tdesla = maxthr*self.ctdesa*np.logical_and.reduce([self.descent, low, (self.phase==4)])
        # phase landing
        Tdesll = maxthr*self.ctdesld*np.logical_and.reduce([self.descent, low, (self.phase==5)])
        # phase ground: minimum descent thrust as a first approach
        Tgd = np.minimum.reduce([Tdesh, Tdeslc])*(self.phase==6)   

        # merge all thrust conditions
        T = np.maximum.reduce([Tjc, Ttc, Tpc, Tlvl, Tdesh, Tdeslc, Tdesla, Tdesll, Tgd])


        # vertical speed
        # vertical speed. Note: ISA only ( tISA = 1 )
        # for climbs: reducing factor (reduced climb power) is multiplied
        # cred applies below 0.8*hmax and for climbing aircraft only
        hcred = np.array(self.traf.alt < (self.hmaxact*0.8))
        clh = np.logical_and.reduce([hcred, self.climb])
        cred = self.cred*clh
        cpred = 1-cred*((self.mmax-self.mass)/(self.mmax-self.mmin)) 
        
        vs = (((T-self.D)*self.traf.tas) /(self.mass*g0))*self.ESF*cpred
        
        # switch for given vertical speed avs
        if (self.traf.avs.any()>0) or (self.traf.avs.any()<0):
            # thrust = f(avs)
            T = ((self.traf.avs!=0)*(((self.traf.pilot.vs*self.mass*g0)/     \
                      (self.ESF*np.maximum(self.traf.eps,self.traf.tas)*cpred)) \
                      + self.D)) + ((self.traf.avs==0)*T)
                      
            vs = (self.traf.avs!=0)*self.traf.avs + (self.traf.avs==0)*vs 
        self.traf.vs = vs
        self.Thr = T
            


        # Fuel consumption
        # thrust specific fuel consumption - jet
        # thrust
        etaj = self.cf1*(1.0+(self.traf.tas/kts)/self.cf2)
        # merge
        ej = etaj*self.jet

        # thrust specific fuel consumption - turboprop

        # thrust
        etat = self.cf1*(1.-(self.traf.tas/kts)/self.cf2)*((self.traf.tas/kts)/1000.)
        # merge
        et = etat*self.turbo
        
        # thrust specific fuel consumption for all aircraft
        # eta is given in [kg/(min*kN)] - convert to [kg/(min*N)]            
        eta = np.maximum.reduce([ej, et])/1000.
     
        # nominal fuel flow - (jet & turbo) and piston
        # condition jet,turbo:
        jt = np.maximum.reduce([self.jet, self.turbo])  
        pdf = np.maximum.reduce ([self.piston])
        
        fnomjt = eta*self.Thr*jt
        fnomp = self.cf1*pdf
        # merge
        fnom = fnomjt + fnomp 

        # minimal fuel flow jet, turbo and piston
        fminjt = self.cf3*(1-(self.traf.alt/ft)/self.cf4)*jt
        fminp = self.cf3*pdf
        #merge
        fmin = fminjt + fminp

        # cruise fuel flow jet, turbo and piston
        fcrjt = eta*self.Thr*self.cf5*jt
        fcrp = self.cf1*self.cf5*pdf 
        #merge
        fcr = fcrjt + fcrp

        # approach/landing fuel flow
        fal = np.maximum(fnom, fmin)
        
        # designate each aircraft to its fuelflow           
        # takeoff
        ffto = fnom*(self.phase==1)

        # initial climb
        ffic = fnom*(self.phase==2)/2

        # phase cruise and climb
        cc = np.logical_and.reduce([self.climb, (self.phase==3)])*1
        ffcc = fnom*cc

        # cruise and level
        ffcrl = fcr*lvl

        # descent cruise configuration
        cd2 = np.logical_and.reduce ([self.descent, (self.phase==3)])*1
        ffcd = cd2*fmin

        # approach
        ffap = fal*(self.phase==4)

        # landing
        ffld = fal*(self.phase==5)
        
        # ground
        ffgd = fmin*(self.phase==6)

        # fuel flow for each condition
        self.ff = np.maximum.reduce([ffto, ffic, ffcc, ffcrl, ffcd, ffap, ffld, ffgd])/60. # convert from kg/min to kg/sec

        # update mass
        self.mass = self.mass - self.ff*self.dt # Use fuelflow in kg/min
        
        
        
        # for aircraft on the runway and taxiways we need to know, whether they
        # are prior or after their flight
        self.post_flight = np.where(self.descent, True, self.post_flight)
        
        # when landing, we would like to stop the aircraft.
        self.traf.aspd = np.where((self.traf.alt <0.5)*(self.post_flight), 0.0, self.traf.aspd)        

        # otherwise taxiing will be impossible afterwards
        self.pf_flag = np.where ((self.traf.alt <0.5)*(self.post_flight), False, self.pf_flag)        
        
        
        
        return

    
    def limits(self):
        """FLIGHT ENVELPOE"""        
        # summarize minimum speeds - ac in ground mode might be pushing back
        self.vmin =  (self.phase == 1) * self.vmto + (self.phase == 2) * self.vmic + (self.phase == 3) * self.vmcr + \
        (self.phase == 4) * self.vmap + (self.phase == 5) * self.vmld + (self.phase == 6) * -10.       

        # maximum altitude: hmax/act = MIN[hmo, hmax+gt*(dtemp-ctc1)+gw*(mmax-mact)]
        #                   or hmo if hmx ==0 ()
        # at the moment just ISA atmosphere, dtemp  = 0            
        c1 = self.dtemp - self.ctct1

        # if c1<0: c1 = 0
        # values above 0 remain, values below are replaced through 0
        c1m = np.array(c1<0)*0.00000000000001
        c1def = np.maximum(c1, c1m)

        self.hact = self.hmax+self.gt*c1def+self.gw*(self.mmax-self.mass)
        # if hmax in OPF File ==0: hmaxact = hmo, else minimum(hmo, hmact)       
        self.hmaxact = (self.hmax==0)*self.hmo +(self.hmax !=0)*np.minimum(self.hmo, self.hact)

        # forwarding to tools
        self.traf.limspd, self.traf.limspd_flag, self.traf.limalt, self.traf.limvs, self.traf.limvs_flag = \
        limits(self.traf.pilot.spd, self.traf.limspd, self.traf.gs,self.vmto, self.vmin, \
        self.vmo, self.mmo, self.traf.M, self.traf.alt, self.hmaxact, \
        self.traf.pilot.alt, self.traf.limalt, self.maxthr, self.Thr,self.traf.limvs, \
        self.D, self.traf.tas, self.mass, self.ESF)        
        
        return

    def acceleration(self, simdt):
        # define acceleration: aircraft taxiing and taking off use ground acceleration,
        # others standard acceleration
        ax = ((self.phase==2) + (self.phase==3) + (self.phase==4) + (self.phase==5) ) \
            *np.minimum(abs(self.traf.delspd / max(1e-8,simdt)), self.traf.ax) + \
            ((self.phase==1) + (self.phase==6))*np.minimum(abs(self.traf.delspd \
            / max(1e-8,simdt)), self.gr_acc)

        return ax
        #------------------------------------------------------------------------------
        #DEBUGGING

        #record data 
        # self.log.write(self.dt, str(self.traf.alt[0]), str(self.traf.tas[0]), str(self.D[0]), str(self.T[0]), str(self.ff[0]),  str(self.traf.vs[0]), str(cd[0]))
        # self.log.save()

        # print self.id, self.phase, self.alt/ft, self.tas/kts, self.cas/kts, self.M,  \
        # self.Thr, self.D, self.ff,  cl, cd, self.vs/fpm, self.ESF,self.atrans, maxthr, \
        # self.vmto/kts, self.vmic/kts ,self.vmcr/kts, self.vmap/kts, self.vmld/kts, \
        # CD0f, kf, self.hmaxact
