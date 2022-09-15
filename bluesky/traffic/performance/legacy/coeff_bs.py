""" BlueSky aircraft performance calculations."""
from xml.etree import ElementTree
from math import *
import numpy as np
from bluesky.tools.aero import ft, g0, rho0, kts, lbs, inch, sqft, fpm

from .performance import esf, phases, calclimits, PHASE
import bluesky as bs

# Register settings defaults
bs.settings.set_variable_defaults(perf_path='performance/BS', verbose=False)

class CoeffBS:
    """
    Coefficient class definition : get aircraft-specific coefficients from database
    Created by  : Isabel Metz

    References:

    - D.P. Raymer. Aircraft Design: A Conceptual Approach. AIAA Education Series.
    American Institute of Aeronautics and Astronautics, Inc., Reston, U.S, fifth edition, 2012.
    - R. Babikian. The Historical Fuel Efficiency Characteristics of Regional Aircraft from
    Technological, Operational, and Cost Perspectives. Master's Thesis, Massachusetts
    Institute of Technology, Boston, U.S.
    """

    def convert(self, value, unit):
        factors = {'kg': 1., 't':1000., 'lbs': lbs, 'N': 1., 'W': 1, \
                    'm':1.,'km': 1000., 'inch': inch,'ft': ft, \
                    'sqm': 1., 'sqft': sqft, 'sqin': 0.0254*0.0254 ,\
                    'm/s': 1., 'km/h': 1./3.6, 'kts': kts, 'fpm': fpm, \
                    "kg/s": 1., "kg/m": 1./60., 'mug/J': 0.000001, 'mg/J': 0.001 ,
                    "kW": 1000.,"kN":1000.,
                    "":1.}

        if unit in factors:
            converted = factors[unit] * float(value)

        else:
            converted = float(value)
            if not self.warned:
                print("traf/perf.py convert function: Unit mismatch. Could not find ", unit)
                self.warned = True

        return converted


    def coeff(self):

        # aircraft
        self.atype     = [] # aircraft type
        self.j_ac      = [] # list of all jet aircraft
        self.tp_ac     = [] # list of all turboprop aircraft

        # engine
        self.etype     = [] # jet / turboprop
        self.engines   = [] # engine types avaliable per aircraft type
        self.j_engines = [] # engine types for jet aircraft
        self.tp_engines= [] # engine types for turboprop aircraft
        self.n_eng     = [] # number of engines

        # weights
        self.MTOW      = [] # maximum takeoff weight

        # speeds
        self.max_spd   = [] # maximum CAS
        self.cr_Ma     = [] # nominal cruise Mach at 35000 ft
        self.cr_spd    = [] # cruise speed
        self.max_Ma    = [] # maximum Mach
        self.gr_acc    = [] # ground acceleration
        self.gr_dec    = [] # ground deceleration

        # limits
        self.vmto      = [] # minimum speed during takeoff
        self.vmld      = [] # minimum speed during landing
        self.clmax_cr  = [] # max. cruise lift coefficient
        self.max_alt   = [] # maximum altitude

        # dimensions
        #span          = [] # wing span
        self.Sref      = [] # reference wing area
        #wet_area      = [] # wetted area

        # aerodynamics
        #Cfe           = [] # equivalent skin friction coefficient (Raymer, p.428)
        self.CD0       = [] # parasite drag coefficient
        #oswald        = [] # oswald factor
        self.k         = [] # induced drag factor

        # scaling factors for drag (FAA_2005 SAGE)
        # order of flight phases: TO, IC, CR ,AP, LD ,LD gear
        self.d_CD0j    = [1.476, 1.143,1.0, 1.957, 3.601, 1.037]
        self.d_kj      = [1.01, 1.071, 1.0 ,0.992, 0.932, 1.0]
        self.d_CD0t    = [1.220, 1.0, 1.0, 1.279, 1.828, 0.496]
        self.d_kt      = [0.948, 1.0, 1.0, 0.94, 0.916, 1.0]

        # bank angles per phase. Order: TO, IC, CR, AP, LD. Currently already in CTraffic
        # self.bank = np.deg2rad(np.array([15,35,35,35,15]))

        # flag: did we already warn about invalid input unit?
        self.warned    = False

        # parse AC files

        path = bs.resource(bs.settings.perf_path) / 'BS/aircraft'
        for fname in path.iterdir():
            acdoc = ElementTree.parse(fname)

            #actype = doc.find('ac_type')
            self.atype.append(acdoc.find('ac_type').text)

            # engine
            self.etype.append(int(acdoc.find('engine/eng_type').text))

            # store jet and turboprop aircraft in seperate lists for accessing specific engine data
            if int(acdoc.find('engine/eng_type').text) ==1:
                self.j_ac.append(acdoc.find('ac_type').text)

            elif int(acdoc.find('engine/eng_type').text) ==2:
                self.tp_ac.append(acdoc.find('ac_type').text)

            self.n_eng.append(float(acdoc.find('engine/num_eng').text))

            engine = []
            for eng in acdoc.findall('engine/eng'):
                engine.append(eng.text)

            # weights
            MTOW = self.convert(acdoc.find('weights/MTOW').text, acdoc.find('weights/MTOW').attrib['unit'])

            self.MTOW.append(MTOW)

            MLW= self.convert(acdoc.find('weights/MLW').text, acdoc.find('weights/MLW').attrib['unit'])

            # dimensions
            # wingspan
            span = self.convert(acdoc.find('dimensions/span').text, acdoc.find('dimensions/span').attrib['unit'])
            # reference surface area
            S_ref = self.convert(acdoc.find('dimensions/wing_area').text, acdoc.find('dimensions/wing_area').attrib['unit'])
            self.Sref.append(S_ref)

            # wetted area
            S_wet = self.convert(acdoc.find('dimensions/wetted_area').text, acdoc.find('dimensions/wetted_area').attrib['unit'])

            # speeds
            # cruise Mach number
            crma = acdoc.find('speeds/cr_MA')
            if float(crma.text) == 0.0:
                # to be refined
                self.cr_Ma.append(0.8)
            else:
                self.cr_Ma.append(float(crma.text))

            # cruise TAS
            crspd = acdoc.find('speeds/cr_spd')

            # to be refined
            if float(crspd.text) == 0.0:
                self.cr_spd.append(self.convert(250, 'kts'))
            else:
                self.cr_spd.append(self.convert(acdoc.find('speeds/cr_spd').text, acdoc.find('speeds/cr_spd').attrib['unit']))

            # ground acceleration
            # values are based on statistical ADS-B evaluations
            # turboprops: 2.12 m/s^2 acceleration,1.12m/s^2 deceleration
            if int(acdoc.find('engine/eng_type').text) == 2:
                self.gr_acc.append(2.12)
                self.gr_dec.append(1.12)

            # turbofans
            else:

                # turbofans with two engines: 1.94 m/^2, 1.265m/s^2 deceleration
                if float(acdoc.find('engine/num_eng').text) == 2. :
                    self.gr_acc.append(1.94)
                    self.gr_dec.append(1.265)
                # turbofans with four engines: 1.68 m/s^2, 1.131 m/s^2 deceleration
                #  assumption: aircraft with three engines have the same value
                else :
                    self.gr_acc.append(1.68)
                    self.gr_dec.append(1.131)



            # limits
            # min takeoff speed
            tospd = acdoc.find('speeds/to_spd')
            # no take-off speed given: calculate via cl_max
            if float (tospd.text) == 0.:
                clmax_to = float(acdoc.find('aerodynamics/clmax_to').text)
                self.vmto.append (sqrt((2*g0)/(S_ref*clmax_to))) # influence of current weight and density follows in CTraffic
            else:
                tospd = self.convert(acdoc.find('speeds/to_spd').text, acdoc.find('speeds/to_spd').attrib['unit'])
                self.vmto.append(tospd/(1.13*sqrt(MTOW/rho0))) # min spd according to CS-/FAR-25.107
            # min ic, cr, ap speed
            clmaxcr = (acdoc.find('aerodynamics/clmax_cr'))
            self.clmax_cr.append(float(clmaxcr.text))

            # min landing speed
            ldspd = acdoc.find('speeds/ld_spd')
            if float(ldspd.text) == 0. :
                clmax_ld = (acdoc.find('aerodynamics/clmax_ld'))
                self.vmld.append (sqrt((2*g0)/(S_ref*float(clmax_ld.text)))) # influence of current weight and density follows in CTraffic
            else:
                ldspd = self.convert(acdoc.find('speeds/ld_spd').text, acdoc.find('speeds/ld_spd').attrib['unit'])
                clmax_ld = MLW*g0*2/(rho0*(ldspd*ldspd)*S_ref)
                self.vmld.append(ldspd/(1.23*sqrt(MLW/rho0)))
            # maximum CAS
            maxspd = acdoc.find('limits/max_spd')
            if float(maxspd.text) == 0.0:
                # to be refined
                self.max_spd.append(400.)
            else:
                self.max_spd.append(self.convert(acdoc.find('limits/max_spd').text, acdoc.find('limits/max_spd').attrib['unit']))
            # maximum Mach
            maxma = acdoc.find('limits/max_MA')
            if float(maxma.text) == 0.0:
                # to be refined
                self.max_Ma.append(0.8)
            else:
                self.max_Ma.append(float(maxma.text))


            # maximum altitude
            maxalt = acdoc.find('limits/max_alt')
            if float(maxalt.text) == 0.0:
                #to be refined
                self.max_alt.append(11000.)
            else:
                self.max_alt.append(self.convert(acdoc.find('limits/max_alt').text, acdoc.find('limits/max_alt').attrib['unit']))

            # aerodynamics

            # parasitic drag - according to Raymer, p. 429
            Cfe = float((acdoc.find('aerodynamics/Cfe').text))
            self.CD0.append (Cfe*S_wet/S_ref)

            # induced drag
            oswald = acdoc.find('aerodynamics/oswald')
            if float(oswald.text) == 0.0:
                # math method according to Obert 2009, p.542: e = 1/(1.02+0.09*pi*AR) combined with Nita 2012, p.2
                self.k.append(1.02/(pi*(span*span/S_ref))+0.009)
            else:
                oswald = float(acdoc.find('aerodynamics/oswald').text)
                self.k.append(1/(pi*oswald*(span*span/S_ref)))

            #users = doc.find( 'engine' )
            #for node in users.getiterator():
            #    print node.tag, node.attrib, node.text, node.tail

            # to collect avaliable engine types per aircraft
            # 2do!!! access via console so user may choose preferred engine
            # for data file: statistics provided by flightglobal for first choice
            # if not declared differently: first engine is taken!
            self.engines.append(engine)

            if int(acdoc.find('engine/eng_type').text) ==1:
                self.j_engines.append(engine)

            elif int(acdoc.find('engine/eng_type').text) ==2:
                self.tp_engines.append(engine)

        # engines
        self.enlist      = [] # list of all engines
        self.jetenlist   = [] # list of all jet engines
        self.propenlist  = [] # list of all turbopropengines

        # a. jet aircraft
        self.rated_thrust = [] # rated Thrust (one engine)
        self.ffto        = [] # fuel flow takeoff
        self.ffcl        = [] # fuel flow climb
        self.ffcr        = [] # fuel flow cruise
        self.ffid        = [] # fuel flow idle
        self.ffap        = [] # fuel flow approach
        self.SFC         = [] # specific fuel flow cruise


        # b. turboprops
        self.P           = [] # max. power (Turboprops, one engine)
        self.PSFC_TO     = [] # SFC takeoff
        self.PSFC_CR     = [] # SFC cruise

        # parse engine files
        path = bs.resource(bs.settings.perf_path) / 'BS/engines/'
        for fname in path.iterdir():
            endoc = ElementTree.parse(fname)
            self.enlist.append(endoc.find('engines/engine').text)

            # thrust
            # a. jet engines
            if int(endoc.find('engines/eng_type').text) ==1:

                # store engine in jet-engine list
                self.jetenlist.append(endoc.find('engines/engine').text)
                # thrust
                self.rated_thrust.append(self.convert(endoc.find('engines/Thr').text, endoc.find('engines/Thr').attrib['unit']))
                # bypass ratio
                BPRc = int(endoc.find('engines/BPR_cat').text)
                # different SFC for different bypass ratios (reference: Raymer, p.36)
                SFC = [14.1, 22.7, 25.5]
                self.SFC.append(SFC[BPRc])

                # fuel flow: Takeoff, climb, cruise, approach, idle
                self.ffto.append(self.convert(endoc.find('ff/ff_to').text, endoc.find('ff/ff_to').attrib['unit']))
                self.ffcl.append(self.convert(endoc.find('ff/ff_cl').text, endoc.find('ff/ff_cl').attrib['unit']))
                self.ffcr.append(self.convert(endoc.find('ff/ff_cr').text, endoc.find('ff/ff_cr').attrib['unit']))
                self.ffap.append(self.convert(endoc.find('ff/ff_ap').text, endoc.find('ff/ff_ap').attrib['unit']))
                self.ffid.append(self.convert(endoc.find('ff/ff_id').text, endoc.find('ff/ff_id').attrib['unit']))

            # b. turboprop engines
            elif int(endoc.find('engines/eng_type').text) ==2:

                # store engine in prop-engine list
                self.propenlist.append(endoc.find('engines/engine').text)

                # power
                self.P.append(self.convert(endoc.find('engines/Power').text, endoc.find('engines/Power').attrib['unit']))
                # specific fuel consumption: takeoff and cruise
                PSFC_TO = self.convert(endoc.find('SFC/SFC_TO').text, endoc.find('SFC/SFC_TO').attrib['unit'])
                self.PSFC_TO.append(PSFC_TO)
                # according to Babikian (function based on PSFC in [mug/J]), input in [kg/J]
                self.PSFC_CR.append(self.convert((0.7675*PSFC_TO*1000000.0 + 23.576), 'mug/J'))
                # print PSFC_TO, self.PSFC_CR

        # Turn relevant ones into numpy arrays

        self.MTOW=np.array(self.MTOW)
        self.Sref=np.array(self.Sref)
        self.etype=np.array(self.etype)
        self.cr_Ma=np.array(self.cr_Ma)
        self.cr_spd=np.array(self.cr_spd)
        self.gr_acc=np.array(self.gr_acc)
        self.gr_dec=np.array(self.gr_dec)
        self.vmto=np.array(self.vmto)
        self.vmld=np.array(self.vmld)
        self.max_Ma=np.array(self.max_Ma)
        self.max_spd=np.array(self.max_spd)
        self.max_alt=np.array(self.max_alt)
        self.CD0=np.array(self.CD0)
        self.k=np.array(self.k)
        self.clmax_cr=np.array(self.clmax_cr)
        self.n_eng=np.array(self.n_eng)
        self.P=np.array(self.P)
        self.PSFC_TO=np.array(self.PSFC_TO)
        self.PSFC_CR=np.array(self.PSFC_CR)
        self.rated_thrust=np.array(self.rated_thrust)
        self.SFC=np.array(self.SFC)
        self.ffto=np.array(self.ffto)
        self.ffcl=np.array(self.ffcl)
        self.ffcr=np.array(self.ffcr)
        self.ffid=np.array(self.ffid)
        self.ffap=np.array(self.ffap)
