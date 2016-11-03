import os
import numpy as np
from xml.etree import ElementTree
from math import *
from ..tools.aero import ft, g0, a0, T0, rho0, gamma1, gamma2,  beta, R, \
    kts, lbs, inch, sqft, fpm, vtas2cas

from performance import esf, phases, limits
from ..settings import data_path

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

    def __init__(self):
        return

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
                print "traf/perf.py convert function: Unit mismatch. Could not find ", unit     
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
                
        path = data_path + '/coefficients/BS_aircraft/'
        files = os.listdir(path)
        for file in files:
            acdoc = ElementTree.parse(path + file)

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
        self.rThr        = [] # rated Thrust (one engine)
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
        path = data_path + '/coefficients/BS_engines/'
        files = os.listdir(path)
        for filename in files:
            endoc = ElementTree.parse(path + filename)
            self.enlist.append(endoc.find('engines/engine').text)

            # thrust
            # a. jet engines            
            if int(endoc.find('engines/eng_type').text) ==1:
                
                # store engine in jet-engine list
                self.jetenlist.append(endoc.find('engines/engine').text)    
                # thrust           
                self.rThr.append(self.convert(endoc.find('engines/Thr').text, endoc.find('engines/Thr').attrib['unit']))
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
        return


coeffBS = CoeffBS()


class Perf():
    warned  = False        # Flag: Did we warn for default perf parameters yet?
    warned2 = False    # Flag: Use of piston engine aircraft?

    def __init__(self, traf):
        # assign needed data from CTraffic
        self.traf = traf

        # create empty database
        self.reset()

        # prepare for coefficient readin
        coeffBS.coeff()

        # Flight performance scheduling
        self.dt  = 0.1           # [s] update interval of performance limits
        self.t0  = -self.dt  # [s] last time checked (in terms of simt)
        self.warned2 = False        # Flag: Did we warn for default engine parameters yet?

        return

    def reset(self):
        """Reset database"""
        # list of aircraft indices
        self.coeffidxlist = np.array([])

        # geometry and weight
        self.mass         = np.array ([])
        self.Sref         = np.array ([])               
        
        # speeds         

        # reference velocities
        self.refma        = np.array([]) # reference Mach
        self.refcas       = np.array([]) # reference CAS  
        self.gr_acc       = np.array([]) # ground acceleration
        self.gr_dec       = np.array([]) # ground deceleration
        
        # limits
        self.vm_to        = np.array([]) # min takeoff spd (w/o mass, density)
        self.vm_ld        = np.array([]) # min landing spd (w/o mass, density) 
        self.vmto         = np.array([]) # min TO spd
        self.vmic         = np.array([]) # min. IC speed
        self.vmcr         = np.array([]) # min cruise spd
        self.vmap         = np.array([]) # min approach speed
        self.vmld         = np.array([]) # min landing spd     
        self.vmin         = np.array([]) # min speed over all phases          
        self.vmo          = np.array([]) # max CAS
        self.mmo          = np.array([]) # max Mach    
        
        self.hmaxact      = np.array([]) # max. altitude
        self.maxthr       = np.array([]) # maximum thrust

        # aerodynamics
        self.CD0          = np.array([]) # parasite drag coefficient
        self.k            = np.array([]) # induced drag factor      
        self.clmaxcr      = np.array([]) # max. cruise lift coefficient
        self.qS           = np.array([])
        
        # engines
        self.traf.engines = [] # avaliable engine type per aircraft type
        self.etype        = np.array([]) # jet /turboprop
        
        # jet engines:
        self.rThr         = np.array([]) # rated thrust (all engines)
        self.Thr_s        = np.array([]) # chosen thrust setting
        self.SFC          = np.array([]) # specific fuel consumption in cruise
        self.ff           = np.array([]) # fuel flow
        self.ffto         = np.array([]) # fuel flow takeoff
        self.ffcl         = np.array([]) # fuel flow climb
        self.ffcr         = np.array([]) # fuel flow cruise
        self.ffid         = np.array([]) # fuel flow idle
        self.ffap         = np.array([]) # fuel flow approach
        self.Thr_s        = np.array([1., 0.85, 0.07, 0.3 ]) # Thrust settings per flight phase according to ICAO

        # turboprop engines
        self.P            = np.array([]) # avaliable power at takeoff conditions
        self.PSFC_TO      = np.array([]) # specific fuel consumption takeoff
        self.PSFC_CR      = np.array([]) # specific fuel consumption cruise
        self.eta          = 0.8          # propeller efficiency according to Raymer
        
        self.Thr          = np.array([]) # Thrust
        self.D            = np.array([]) # Drag
        self.ESF          = np.array([]) # Energy share factor according to EUROCONTROL
        
        # flight phase
        self.phase        = np.array([]) # flight phase
        self.bank         = np.array([]) # bank angle    
        self.post_flight  = np.array([]) # check for ground mode: 
                                          #taxi prior of after flight
        self.pf_flag      = np.array([])
        return
       

    def create(self):
        actype = self.traf.type[-1]
        """Create new aircraft"""
        # note: coefficients are initialized in SI units
        if actype in coeffBS.atype:
            # aircraft
            self.coeffidx = coeffBS.atype.index(actype)
            # engine
        else:
            self.coeffidx = 0
            if not Perf.warned:
                  print "aircraft is using default aircraft performance (Boeing 747-400)."
            Perf.warned = True
        self.coeffidxlist = np.append(self.coeffidxlist, self.coeffidx)
        self.mass         = np.append(self.mass, coeffBS.MTOW[self.coeffidx]) # aircraft weight
        self.Sref         = np.append(self.Sref, coeffBS.Sref[self.coeffidx]) # wing surface reference area
        self.etype        = np.append(self.etype, coeffBS.etype[self.coeffidx]) # engine type of current aircraft
        self.traf.engines.append(coeffBS.engines[self.coeffidx]) # avaliable engine type per aircraft type   

        # speeds             
        self.refma        = np.append(self.refma, coeffBS.cr_Ma[self.coeffidx]) # nominal cruise Mach at 35000 ft
        self.refcas       = np.append(self.refcas, vtas2cas(coeffBS.cr_spd[self.coeffidx], 35000*ft)) # nominal cruise CAS
        self.gr_acc       = np.append(self.gr_acc,coeffBS.gr_acc[self.coeffidx]) # ground acceleration
        self.gr_dec       = np.append(self.gr_dec, coeffBS.gr_dec[self.coeffidx]) # ground acceleration
        
        # calculate the crossover altitude according to the BADA 3.12 User Manual
        self.atrans       = ((1000/6.5)*(T0*(1-((((1+gamma1*(self.refcas/a0)*(self.refcas/a0))** \
                                (gamma2))-1) / (((1+gamma1*self.refma*self.refma)** \
                                    (gamma2))-1))**((-(beta)*R)/g0))))

        # limits   
        self.vm_to        = np.append(self.vm_to, coeffBS.vmto[self.coeffidx])
        self.vm_ld        = np.append(self.vm_ld, coeffBS.vmld[self.coeffidx])   
        self.vmto         = np.append(self.vmto, 0.0)
        self.vmic         = np.append(self.vmic, 0.0)        
        self.vmcr         = np.append(self.vmcr, 0.0)
        self.vmap         = np.append(self.vmap, 0.0)
        self.vmld         = np.append(self.vmld, 0.0)
        self.vmin         = np.append (self.vmin, 0.0)
        self.mmo          = np.append(self.mmo, coeffBS.max_Ma[self.coeffidx]) # maximum Mach
        self.vmo          = np.append(self.vmo, coeffBS.max_spd[self.coeffidx]) # maximum CAS
        self.hmaxact      = np.append(self.hmaxact, coeffBS.max_alt[self.coeffidx]) # maximum altitude  
        
        # aerodynamics
        self.CD0          = np.append(self.CD0, coeffBS.CD0[self.coeffidx])  # parasite drag coefficient
        self.k            = np.append(self.k, coeffBS.k[self.coeffidx])  # induced drag factor   
        self.clmaxcr      = np.append(self.clmaxcr, coeffBS.clmax_cr[self.coeffidx])   # max. cruise lift coefficient
        self.qS           = np.append(self.qS, 0.0)
        # performance - initialise neutrally       
        self.D            = np.append(self.D, 0.) 
        self.ESF          = np.append(self.ESF, 1.)
        
        # flight phase
        self.phase        = np.append(self.phase, 0.)
        self.bank         = np.append(self.bank, 0.)
        self.post_flight  = np.append(self.post_flight, False) # for initialisation,
                                                              # we assume that ac has yet to take off
        self.pf_flag      = np.append(self.pf_flag, True)
        # engines

        # turboprops
        if coeffBS.etype[self.coeffidx] ==2:
            if coeffBS.engines[self.coeffidx][0] in coeffBS.propenlist:
                self.propengidx = coeffBS.propenlist.index(coeffBS.engines[self.coeffidx][0])
            else:
                self.propengidx = 0
                if not Perf.warned2:
                    print "prop aircraft is using standard engine. Please check valid engine types per aircraft type"
                    Perf.warned2 = True

            self.P       = np.append(self.P, coeffBS.P[self.propengidx]*coeffBS.n_eng[self.coeffidx])                     
            self.PSFC_TO = np.append(self.PSFC_TO, coeffBS.PSFC_TO[self.propengidx]) 
            self.PSFC_CR = np.append(self.PSFC_CR, coeffBS.PSFC_CR[self.propengidx])
            self.ff      = np.append(self.ff, 0.) # neutral initialisation            
            # jet characteristics needed for numpy calculations
            self.rThr    = np.append(self.rThr, 1.) 
            self.Thr     = np.append(self.Thr, 1.)        
            self.maxthr  = np.append (self.maxthr, 1.) 
            self.SFC     = np.append(self.SFC, 1.)
            self.ffto    = np.append(self.ffto, 1.)
            self.ffcl    = np.append(self.ffcl, 1.)
            self.ffcr    = np.append(self.ffcr, 1.)
            self.ffid    = np.append(self.ffid, 1.)
            self.ffap    = np.append(self.ffap, 1.)


        # jet (also default)

        else:      # so coeffBS.etype[self.coeffidx] ==1:

            if coeffBS.engines[self.coeffidx][0] in coeffBS.jetenlist:
                self.jetengidx = coeffBS.jetenlist.index(coeffBS.engines[self.coeffidx][0])
            else:
                self.jetengidx = 0
                if not self.warned2:
                    print " jet aircraft is using standard engine. Please check valid engine types per aircraft type"
                    self.warned2 = True

            self.rThr    = np.append(self.rThr, coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[self.coeffidx])  # rated thrust (all engines)
            self.Thr     = np.append(self.Thr, coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[self.coeffidx])  # initialize thrust with rated thrust       
            self.maxthr  = np.append (self.maxthr, coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[self.coeffidx]*1.2)  # maximum thrust - initialize with 1.2*rThr
            self.SFC     = np.append(self.SFC, coeffBS.SFC[self.jetengidx])
            self.ff      = np.append(self.ff, 0.)  # neutral initialisation
            self.ffto    = np.append(self.ffto, coeffBS.ffto[self.jetengidx]*coeffBS.n_eng[self.coeffidx])
            self.ffcl    = np.append(self.ffcl, coeffBS.ffcl[self.jetengidx]*coeffBS.n_eng[self.coeffidx])
            self.ffcr    = np.append(self.ffcr, coeffBS.ffcr[self.jetengidx]*coeffBS.n_eng[self.coeffidx])
            self.ffid    = np.append(self.ffid, coeffBS.ffid[self.jetengidx]*coeffBS.n_eng[self.coeffidx])
            self.ffap    = np.append(self.ffap, coeffBS.ffap[self.jetengidx]*coeffBS.n_eng[self.coeffidx])

            # propeller characteristics needed for numpy calculations
            self.P       = np.append(self.P, 1.)
            self.PSFC_TO = np.append(self.PSFC_TO, 1.)
            self.PSFC_CR = np.append(self.PSFC_CR, 1.)

        return

    def delete(self, idx):
        """Delete removed aircraft"""

        del self.traf.engines[idx]

        self.coeffidxlist = np.delete(self.coeffidxlist, idx)
        self.mass         = np.delete(self.mass, idx)    # aircraft weight
        self.Sref         = np.delete(self.Sref, idx)    # wing surface reference area
        self.etype        = np.delete(self.etype, idx)  # engine type of current aircraft

        # limits
        self.vmo          = np.delete(self.vmo, idx)  # maximum CAS
        self.mmo          = np.delete(self.mmo, idx)  # maximum Mach

        # vm_to excludes (squrt(m/rho) )
        # vmto includes weight and altitude influence
        self.vm_to        = np.delete(self.vm_to, idx)
        self.vm_ld        = np.delete(self.vm_ld, idx)
        self.vmto         = np.delete(self.vmto, idx)
        self.vmic         = np.delete(self.vmic, idx)
        self.vmcr         = np.delete(self.vmcr, idx)
        self.vmap         = np.delete(self.vmap, idx)
        self.vmld         = np.delete(self.vmld, idx)
        self.vmin         = np.delete(self.vmin, idx)
        self.maxthr       = np.delete(self.maxthr, idx)
        self.hmaxact      = np.delete(self.hmaxact, idx)

        # reference speeds
        self.refma        = np.delete(self.refma, idx)   # nominal cruise Mach at 35000 ft
        self.refcas       = np.delete(self.refcas, idx)  # nominal cruise CAS
        self.gr_acc       = np.delete(self.gr_acc, idx)       # ground acceleration
        self.gr_dec       = np.delete(self.gr_dec, idx)       # ground deceleration
        self.atrans       = np.delete(self.atrans, idx)   # crossover altitude

        # aerodynamics
        self.CD0          = np.delete(self.CD0, idx)      # parasite drag coefficient
        self.k            = np.delete(self.k, idx)        # induced drag factor
        self.clmaxcr      = np.delete(self.clmaxcr, idx)  # max. cruise lift coefficient
        self.qS           = np.delete(self.qS, idx)

        # engine
        self.rThr         = np.delete(self.rThr, idx)     # rated thrust (all engines)
        self.SFC          = np.delete(self.SFC, idx)
        self.ffto         = np.delete(self.ffto, idx)
        self.ffcl         = np.delete(self.ffcl, idx)
        self.ffcr         = np.delete(self.ffcr, idx)
        self.ffid         = np.delete(self.ffid, idx)
        self.ffap         = np.delete(self.ffap, idx)
        self.ff           = np.delete(self.ff, idx)

        # turboprop engines
        self.P            = np.delete(self.P, idx)        # avaliable power at takeoff conditions
        self.PSFC_TO      = np.delete(self.PSFC_TO, idx)  # specific fuel consumption takeoff
        self.PSFC_CR      = np.delete(self.PSFC_CR, idx)  # specific fuel consumption cruise

        # performance
        self.Thr          = np.delete(self.Thr, idx)
        self.D            = np.delete(self.D, idx)
        self.ESF          = np.delete(self.ESF, idx)

        # flight phase
        self.phase        = np.delete(self.phase, idx)
        self.bank         = np.delete(self.bank, idx)
        self.post_flight  = np.delete(self.post_flight, idx)
        self.pf_flag      = np.delete(self.pf_flag, idx)

        return

    def perf(self,simt):
        if abs(simt - self.t0) >= self.dt:
            self.t0 = simt
        else:
            return
        """Aircraft performance"""
        swbada = False # no-bada version

        # allocate aircraft to their flight phase
        self.phase, self.bank = \
           phases(self.traf.alt, self.traf.gs, self.traf.delalt, \
           self.traf.cas, self.vmto, self.vmic, self.vmap, self.vmcr, self.vmld, self.traf.bank, self.traf.bphase, \
           self.traf.hdgsel,swbada)

        # AERODYNAMICS
        # compute CL: CL = 2*m*g/(VTAS^2*rho*S)
        self.qS = 0.5*self.traf.rho*np.maximum(1.,self.traf.tas)*np.maximum(1.,self.traf.tas)*self.Sref

        cl = self.mass*g0/(self.qS*np.cos(self.bank))*(self.phase!=6)+ 0.*(self.phase==6)

        # scaling factors for CD0 and CDi during flight phases according to FAA (2005): SAGE, V. 1.5, Technical Manual
        
        CD0f = (self.phase==1)*(self.etype==1)*coeffBS.d_CD0j[0] + \
               (self.phase==2)*(self.etype==1)*coeffBS.d_CD0j[1]  + \
               (self.phase==3)*(self.etype==1)*coeffBS.d_CD0j[2] + \
               (self.phase==4)*(self.etype==1)*coeffBS.d_CD0j[3] + \
               (self.phase==5)*(self.etype==1)*(self.traf.alt>=450)*coeffBS.d_CD0j[4] + \
               (self.phase==5)*(self.etype==1)*(self.traf.alt<450)*coeffBS.d_CD0j[5] + \
               (self.phase==1)*(self.etype==2)*coeffBS.d_CD0t[0] + \
               (self.phase==2)*(self.etype==2)*coeffBS.d_CD0t[1]  + \
               (self.phase==3)*(self.etype==2)*coeffBS.d_CD0t[2] + \
               (self.phase==4)*(self.etype==2)*coeffBS.d_CD0t[3]
                   # (self.phase==5)*(self.etype==2)*(self.alt>=450)*coeffBS.d_CD0t[4] + \
                   # (self.phase==5)*(self.etype==2)*(self.alt<450)*coeffBS.d_CD0t[5]
                   
        kf =   (self.phase==1)*(self.etype==1)*coeffBS.d_kj[0] + \
               (self.phase==2)*(self.etype==1)*coeffBS.d_kj[1]  + \
               (self.phase==3)*(self.etype==1)*coeffBS.d_kj[2] + \
               (self.phase==4)*(self.etype==1)*coeffBS.d_kj[3] + \
               (self.phase==5)*(self.etype==1)*(self.traf.alt>=450)*coeffBS.d_kj[4] + \
               (self.phase==5)*(self.etype==1)*(self.traf.alt<450)*coeffBS.d_kj[5] + \
               (self.phase==1)*(self.etype==2)*coeffBS.d_kt[0] + \
               (self.phase==2)*(self.etype==2)*coeffBS.d_kt[1]  + \
               (self.phase==3)*(self.etype==2)*coeffBS.d_kt[2] + \
               (self.phase==4)*(self.etype==2)*coeffBS.d_kt[3] + \
               (self.phase==5)*(self.etype==2)*(self.traf.alt>=450)*coeffBS.d_kt[4] + \
               (self.phase==5)*(self.etype==2)*(self.traf.alt<450)*coeffBS.d_kt[5]   


        # drag coefficient
        cd = self.CD0*CD0f + self.k*kf*(cl*cl)

        # compute drag: CD = CD0 + CDi * CL^2 and D = rho/2*VTAS^2*CD*S
        self.D = cd*self.qS

        # energy share factor and crossover altitude  
        epsalt = np.array([0.001]*self.traf.ntraf)   
        self.climb = np.array(self.traf.delalt > epsalt)
        self.descent = np.array(self.traf.delalt< -epsalt)
  

        # crossover altitiude
        self.traf.abco = np.array(self.traf.alt>self.atrans)
        self.traf.belco = np.array(self.traf.alt<self.atrans)

        # energy share factor
        self.ESF = esf(self.traf.abco, self.traf.belco, self.traf.alt, self.traf.M,\
                  self.climb, self.descent, self.traf.delspd)


        # determine vertical speed
        swvs = (np.abs(self.traf.pilot.spd) > self.traf.eps)
        vspd = swvs * self.traf.pilot.spd + (1. - swvs) * self.traf.avs * np.sign(self.traf.delalt)
        swaltsel = np.abs(self.traf.delalt) > np.abs(2. * self.dt * np.abs(vspd))
        self.traf.vs = swaltsel * vspd  

        # determine thrust
        self.Thr = (((self.traf.vs*self.mass*g0)/(self.ESF*np.maximum(self.traf.eps, self.traf.tas))) + self.D) 

        # maximum thrust jet (Bruenig et al., p. 66): 
        mt_jet = self.rThr*(self.traf.rho/rho0)**0.75

        # maximum thrust prop (Raymer, p.36):
        mt_prop = self.P*self.eta/np.maximum(self.traf.eps, self.traf.tas)

        # merge
        self.maxthr = mt_jet*(self.etype==1) + mt_prop*(self.etype==2)

        # Fuel Flow
        
        # jet aircraft
        # ratio current thrust/rated thrust 
        pThr = self.Thr/self.rThr 
        # fuel flow is assumed to be proportional to thrust(Torenbeek, p.62). 
        #For ground operations, idle thrust is used
        # cruise thrust is approximately equal to approach thrust
        ff_jet = ((pThr*self.ffto)*(self.phase!=6)*(self.phase!=3)+ \
        self.ffid*(self.phase==6) + self.ffap*(self.phase==3) )*(self.etype==1)  
        # print "FFJET",  (pThr*self.ffto)*(self.phase!=6)*(self.phase!=3), self.ffid*(self.phase==6), self.ffap*(self.phase==3)   
        # print "FFJET", ff_jet

        # turboprop aircraft
        # to be refined - f(spd)
        # CRUISE-ALTITUDE!!!
        # above cruise altitude: PSFC_CR
        PSFC = (((self.PSFC_CR - self.PSFC_TO) / 20000.0)*self.traf.alt + self.PSFC_TO)*(self.traf.alt<20.000) + \
                self.PSFC_CR*(self.traf.alt >= 20.000)

        TSFC =PSFC*self.traf.tas/(550.0*self.eta)

        # formula p.36 Raymer is missing here!
        ff_prop = self.Thr*TSFC*(self.etype==2)


        # combine
        self.ff = ff_jet + ff_prop

        # update mass
        #self.mass = self.mass - self.ff*self.dt/60. # Use fuelflow in kg/min

        # print self.traf.id, self.phase, self.traf.alt/ft, self.traf.tas/kts, self.traf.cas/kts, self.traf.M,  \
        # self.Thr, self.D, self.ff,  cl, cd, self.traf.vs/fpm, self.ESF,self.atrans, self.maxthr, \
        # self.vmto/kts, self.vmic/kts ,self.vmcr/kts, self.vmap/kts, self.vmld/kts, \
        # CD0f, kf, self.hmaxact
        

        # for aircraft on the runway and taxiways we need to know, whether they
        # are prior or after their flight
        self.post_flight = np.where(self.descent, True, self.post_flight)
        
        # when landing, we would like to stop the aircraft.
        self.traf.aspd = np.where((self.traf.alt <0.5)*(self.post_flight)*self.pf_flag, 0.0, self.traf.aspd)
        # the impulse for reducing the speed to 0 should only be given once,
        # otherwise taxiing will be impossible afterwards
        self.pf_flag = np.where ((self.traf.alt <0.5)*(self.post_flight), False, self.pf_flag)

        return

    def limits(self):
        """Flight envelope""" # Connect this with function limits in performance.py

        # combine minimum speeds and flight phases. Phases initial climb, cruise
        # and approach use the same CLmax and thus the same function for Vmin
        self.vmto = self.vm_to*np.sqrt(self.mass/self.traf.rho)
        self.vmic = np.sqrt(2*self.mass*g0/(self.traf.rho*self.clmaxcr*self.Sref))
        self.vmcr = self.vmic
        self.vmap = self.vmic
        self.vmld = self.vm_ld*np.sqrt(self.mass/self.traf.rho)

        # summarize and convert to cas
        # note: aircraft on ground may be pushed back
        self.vmin = (self.phase==1)*vtas2cas(self.vmto, self.traf.alt) + \
                        ((self.phase==2) + (self.phase==3) + (self.phase==4))*vtas2cas(self.vmcr, self.traf.alt) + \
                            (self.phase==5)*vtas2cas(self.vmld, self.traf.alt) + (self.phase==6)*-10.0


        # forwarding to tools
        self.traf.limspd,          \
        self.traf.limspd_flag,     \
        self.traf.limalt,          \
        self.traf.limvs,           \
        self.traf.limvs_flag  =  limits(self.traf.pilot.spd,   \
                                        self.traf.limspd,      \
                                        self.traf.gs,          \
                                        self.vmto,             \
                                        self.vmin,             \
                                        self.vmo,              \
                                        self.mmo,              \
                                        self.traf.M,           \
                                        self.traf.alt,         \
                                        self.hmaxact,          \
                                        self.traf.pilot.alt,   \
                                        self.traf.limalt,      \
                                        self.maxthr,           \
                                        self.Thr,              \
                                        self.traf.limvs,       \
                                        self.D,                \
                                        self.traf.tas,         \
                                        self.mass,             \
                                        self.ESF)        


        return

    def acceleration(self, simdt):
        # define acceleration: aircraft taxiing and taking off use ground acceleration,
        # landing aircraft use ground deceleration, others use standard acceleration
        ax = ((self.phase==2) + (self.phase==3) + (self.phase==4) + (self.phase==5) ) \
            *np.minimum(abs(self.traf.delspd / max(1e-8,simdt)), self.traf.ax) + \
            ((self.phase==1) + (self.phase==6)*(1-self.post_flight))*np.minimum(abs(self.traf.delspd \
            / max(1e-8,simdt)), self.gr_acc) + \
            (self.phase==6)*self.post_flight*np.minimum(abs(self.traf.delspd \
            / max(1e-8,simdt)), self.gr_dec)
        return ax
    
        
    def engchange(self, idx, engid=None):
        """change of engines - for jet aircraft only!"""
        if not engid:
            disptxt = "available engine types:\n" + '\n'.join(self.traf.engines[idx]) + \
                      "\nChange engine with ENG acid engine_id"
            return False, disptxt
        engidx = self.traf.engines[idx].index(engid)
        self.jetengidx = coeffBS.jetenlist.index(coeffBS.engines[idx][engidx])

        # exchange engine parameters

        self.rThr[idx]   = coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[idx] # rated thrust (all engines)
        self.Thr[idx]    = coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[idx] # initialize thrust with rated thrust       
        self.maxthr[idx] = coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[idx] # maximum thrust - initialize with 1.2*rThr
        self.SFC[idx]    = coeffBS.SFC[self.jetengidx]
        self.ff[idx]     = 0. # neutral initialisation
        self.ffto[idx]   = coeffBS.ffto[self.jetengidx]*coeffBS.n_eng[idx]
        self.ffcl[idx]   = coeffBS.ffcl[self.jetengidx]*coeffBS.n_eng[idx]
        self.ffcr[idx]   = coeffBS.ffcr[self.jetengidx]*coeffBS.n_eng[idx] 
        self.ffid[idx]   = coeffBS.ffid[self.jetengidx]*coeffBS.n_eng[idx]
        self.ffap[idx]   = coeffBS.ffap[self.jetengidx]*coeffBS.n_eng[idx]         
        return
