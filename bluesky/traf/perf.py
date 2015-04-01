import numpy as np

from math import *
from ..tools.aero import fpm, kts, ft, nm, g0, a0, T0, rho0, tas2eas, tas2mach, tas2cas, mach2cas,  \
                 eas2tas, temp, density, Rearth, gamma, gamma1, gamma2,  beta, R
from ..tools.aero_np import vatmos, vcas2tas, veas2tas, vtas2cas, vvsound, vtas2mach, vmach2cas, \
                    cas2mach, vmach2tas, qdrdist
                    
from ..tools.performance import esf, phases, limits, vmin
from ..tools.datalog import Datalog

from params import Coefficients, CoeffBS

coeff = Coefficients()
coeffBS = CoeffBS()

class Perf():
    def __init__(self, traf):
        # assign needed data from CTraffic
        self.traf = traf

        self.warned = False        # Flag: Did we warn for default perf parameters yet?
        self.warned2 = False    # Flag: Use of piston engine aircraft?

        # create empty database
        self.reset()

        # prepare for coefficient readin
        coeffBS.coeff()
        return

    def reset(self):
        """Reset database"""
        # list of aircraft indices
        self.coeffidxlist = np.array([])

        # geometry and weight
        self.mass = np.array ([])
        self.Sref = np.array ([])               
        
        # speeds         
        self.to_spd =np.array ([]) # nominal takeoff speed
        self.ld_spd    = np.array([]) # nominal landing speed
        
        # reference velocities
        self.refma= np.array ([]) # reference Mach
        self.refcas= np.array ([]) # reference CAS        
        
        # limits
        self.vm_to = np.array([]) # min takeoff spd (w/o mass, density)
        self.vm_ld = np.array([]) # min landing spd (w/o mass, density) 
        self.vmto = np.array([]) # min TO spd
        self.vmic = np.array([]) # min. IC speed
        self.vmcr = np.array([]) # min cruise spd
        self.vmap = np.array([]) # min approach speed
        self.vmld = np.array([]) # min landing spd     
        self.vmin = np.array([]) # min speed over all phases          
        self.vmo  = np.array ([]) # max CAS
        self.mmo  = np.array ([]) # max Mach    
        
        self.hmaxact = np.array([]) # max. altitude
        self.maxthr = np.array([]) # maximum thrust

        # aerodynamics
        self.CD0       = np.array([])  # parasite drag coefficient
        self.k         = np.array([])  # induced drag factor      
        self.clmaxcr   = np.array([])   # max. cruise lift coefficient
        
        # engines
        self.traf.engines = [] # avaliable engine type per aircraft type
        self.etype = np.array ([]) # jet /turboprop
        
        # jet engines:
        self.rThr = np.array([]) # rated thrust (all engines)
        self.Thr_s = np.array([]) # chosen thrust setting
        self.SFC  = np.array([]) # specific fuel consumption in cruise
        self.ff = np.array([]) # fuel flow
        self.ffto = np.array([]) # fuel flow takeoff
        self.ffcl = np.array([]) # fuel flow climb
        self.ffcr = np.array ([]) # fuel flow cruise
        self.ffid = np.array([]) # fuel flow idle
        self.ffap = np.array([]) # fuel flow approach
        self.Thr_s= np.array([1., 0.85, 0.07, 0.3 ]) # Thrust settings per flight phase according to ICAO

        # turboprop engines
        self.P = np.array([])    # avaliable power at takeoff conditions
        self.PSFC_TO = np.array([]) # specific fuel consumption takeoff
        self.PSFC_CR = np.array([]) # specific fuel consumption cruise
        self.eta = 0.8           # propeller efficiency according to Raymer
        
        self.Thr = np.array([]) # Thrust
        self.D = np.array([]) # Drag
        self.ESF = np.array([]) # Energy share factor according to EUROCONTROL
        
        # flight phase
        self.phase = np.array([]) # flight phase
        self.bank = np.array ([]) # bank angle        
        return
       

    def create(self, actype):
        """Create new aircraft"""
        # note: coefficients are initialized in SI units
        try:
            # aircraft
            self.coeffidx = coeffBS.atype.index(actype)
            print actype
            # engine
        except:
            self.coeffidx = 0
            if not self.warned:
                  print "aircraft is using default aircraft performance (Boeing 747-400)."
            self.warned = True
        self.coeffidxlist = np.append(self.coeffidxlist, self.coeffidx)
        self.mass = np.append(self.mass, coeffBS.MTOW[self.coeffidx]) # aircraft weight
        self.Sref = np.append(self.Sref, coeffBS.Sref[self.coeffidx]) # wing surface reference area
        self.etype = np.append(self.etype, coeffBS.etype[self.coeffidx]) # engine type of current aircraft
        self.traf.engines.append(coeffBS.engines[self.coeffidx]) # avaliable engine type per aircraft type   

        # speeds         
        # self.to_spd    = np.append(self.to_spd, coeffBS.to_spd[self.coeffidx]) # nominal takeoff speed
        # self.ld_spd    = np.append(self.ld_spd, coeffBS.ld_spd[self.coeffidx]) # nominal landing speed            

        self.refma  = np.append(self.refma, coeffBS.cr_Ma[self.coeffidx]) # nominal cruise Mach at 35000 ft
        self.refcas = np.append(self.refcas, vtas2cas(coeffBS.cr_spd[self.coeffidx], 35000*ft)) # nominal cruise CAS

        # limits   
        self.vm_to = np.append(self.vmto, coeffBS.vmto[self.coeffidx])
        self.vm_ld = np.append(self.vmld, coeffBS.vmld[self.coeffidx])   
        self.vmto = np.append(self.vmto, 0.0)
        self.vmic = np.append(self.vmic, 0.0)        
        self.vmcr = np.append(self.vmcr, 0.0)
        self.vmap = np.append(self.vmap, 0.0)
        self.vmld = np.append(self.vmld, 0.0)
        self.vmin = np.append (self.vmin, 0.0)
        self.mmo    = np.append(self.mmo, coeffBS.max_Ma[self.coeffidx]) # maximum Mach
        self.vmo   = np.append(self.vmo, coeffBS.max_spd[self.coeffidx]) # maximum CAS
        self.hmaxact   = np.append(self.hmaxact, coeffBS.max_alt[self.coeffidx]) # maximum altitude  
        
        # aerodynamics
        self.CD0       = np.append(self.CD0, coeffBS.CD0[self.coeffidx])  # parasite drag coefficient
        self.k         = np.append(self.k, coeffBS.k[self.coeffidx])  # induced drag factor   
        self.clmaxcr   = np.append(self.clmaxcr, coeffBS.clmax_cr[self.coeffidx])   # max. cruise lift coefficient

        # performance - initialise neutrally       
        self.D = np.append(self.D, 0.) 
        self.ESF = np.append(self.ESF, 1.)
        
        # flight phase
        self.phase = np.append(self.phase, 0.)
        
        # engines

        # turboprops
        if coeffBS.etype[self.coeffidx] ==2:
            try:
                self.propengidx = coeffBS.propenlist.index(coeffBS.engines[self.coeffidx][0])
            except:
                self.propengidx = 0
                if not self.warned2:
                    print "prop aircraft is using standard engine. Please check valid engine types per aircraft type"
                    self.warned2 = True
                    
            self.P = np.append(self.P, coeffBS.P[self.propengidx]*coeffBS.n_eng[self.coeffidx])                     
            self.PSFC_TO = np.append(self.PSFC_TO, coeffBS.PSFC_TO[self.propengidx]) 
            self.PSFC_CR = np.append(self.PSFC_CR, coeffBS.PSFC_CR[self.propengidx])
            self.ff = np.append(self.ff, 0.) # neutral initialisation            
            # jet characteristics needed for numpy calculations
            self.rThr = np.append(self.rThr, 1.) 
            self.Thr = np.append(self.Thr, 1.)        
            self.maxthr = np.append (self.maxthr, 1.) 
            self.SFC = np.append(self.SFC, 1.)
            self.ffto = np.append(self.ffto, 1.)
            self.ffcl = np.append(self.ffcl, 1.)
            self.ffcr = np.append(self.ffcr, 1.)
            self.ffid = np.append(self.ffid, 1.)
            self.ffap = np.append(self.ffap, 1.)  


        # jet (also default)

        else:      # so coeffBS.etype[self.coeffidx] ==1:

            try:
                self.jetengidx = coeffBS.jetenlist.index(coeffBS.engines[self.coeffidx][0])
            except:
                self.jetengidx = 0
                if not self.warned2:
                    print " jet aircraft is using standard engine. Please check valid engine types per aircraft type"
                    self.warned2 = True        
                    
            self.rThr = np.append(self.rThr, coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[self.coeffidx]) # rated thrust (all engines)
            self.Thr = np.append(self.Thr, coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[self.coeffidx]) # initialize thrust with rated thrust       
            self.maxthr = np.append (self.maxthr, coeffBS.rThr[self.jetengidx]*coeffBS.n_eng[self.coeffidx]*1.2) # maximum thrust - initialize with 1.2*rThr
            self.SFC = np.append(self.SFC, coeffBS.SFC[self.jetengidx])
            self.ff = np.append(self.ff, 0.) # neutral initialisation
            self.ffto = np.append(self.ffto, coeffBS.ffto[self.jetengidx]*coeffBS.n_eng[self.coeffidx])
            self.ffcl = np.append(self.ffcl, coeffBS.ffcl[self.jetengidx]*coeffBS.n_eng[self.coeffidx])
            self.ffcr = np.append(self.ffcr, coeffBS.ffcr[self.jetengidx]*coeffBS.n_eng[self.coeffidx] )
            self.ffid = np.append(self.ffid, coeffBS.ffid[self.jetengidx]*coeffBS.n_eng[self.coeffidx])
            self.ffap = np.append(self.ffap, coeffBS.ffap[self.jetengidx]*coeffBS.n_eng[self.coeffidx])  
            
            # propeller characteristics needed for numpy calculations
            self.P = np.append(self.P, 1.)                     
            self.PSFC_TO = np.append(self.PSFC_TO, 1.) 
            self.PSFC_CR = np.append(self.PSFC_CR, 1.)            
                    
        return

    def delete(self, idx):
        """Delete removed aircraft"""

        del self.traf.engines[idx]        
           
        self.coeffidxlist = np.delete(self.coeffidxlist, idx)
        self.mass = np.delete(self.mass, idx) # aircraft weight
        self.Sref = np.delete(self.Sref, idx) # wing surface reference area
        self.etype = np.delete(self.etype, idx) # engine type of current aircraft  
        
        # limits         
        # self.to_spd    = np.append(self.to_spd, coeffBS.to_spd[self.coeffidx]) # nominal takeoff speed
        # self.ld_spd    = np.append(self.ld_spd, coeffBS.ld_spd[self.coeffidx]) # nominal landing speed            
        self.vmo   = np.delete(self.vmo, idx) # maximum CAS
        self.mmo    = np.delete(self.mmo, idx) # maximum Mach

        self.vm_to = np.delete(self.vm_to, idx)
        self.vm_ld = np.delete(self.vm_ld, idx)
        self.vmto = np.delete(self.vmto, idx)
        self.vmic = np.delete(self.vmic, idx)
        self.vmcr = np.delete(self.vmcr, idx)        
        self.vmap = np.delete(self.vmap, idx)
        self.vmld = np.delete(self.vmld, idx)
        self.vmin = np.delete(self.vmin, idx)
        self.maxthr = np.delete(self.maxthr, idx) 
        self.hmaxact   = np.delete(self.hmaxact, idx) 

        # reference speeds
        self.refma  = np.delete(self.refma, idx) # nominal cruise Mach at 35000 ft
        self.refcas = np.delete(self.refcas, idx) # nominal cruise CAS

        # aerodynamics
        self.CD0       = np.delete(self.CD0, idx)  # parasite drag coefficient
        self.k         = np.delete(self.k, idx)  # induced drag factor   
        self.clmaxcr   = np.delete(self.clmaxcr, idx)   # max. cruise lift coefficient
        self.qS        = np.delete(self.qS, idx)   

        # engine
        self.rThr = np.delete(self.rThr, idx) # rated thrust (all engines)
        self.SFC = np.delete(self.SFC, idx)
        self.ffto = np.delete(self.ffto, idx)
        self.ffcl = np.delete(self.ffcl, idx)
        self.ffcr = np.delete(self.ffcr, idx)
        self.ffid = np.delete(self.ffid, idx)
        self.ffap = np.delete(self.ffap, idx)
        self.ff   = np.delete(self.ff, idx)

        # turboprop engines
        self.P = np.delete(self.P, idx)    # avaliable power at takeoff conditions
        self.PSFC_TO = np.delete(self.PSFC_TO, idx) # specific fuel consumption takeoff
        self.PSFC_CR = np.delete(self.PSFC_CR, idx) # specific fuel consumption cruise

        # performance
        self.Thr = np.delete(self.Thr, idx)
        self.D = np.delete(self.D, idx)       
        self.ESF = np.delete(self.ESF, idx)

        # flight phase
        self.phase = np.delete(self.phase, idx)
        self.bank = np.delete(self.bank, idx)

        return

    def perf(self):
        """Aircraft performance"""
        
        # allocate aircraft to their flight phase
        self.phase, self.bank = \
        phases(self.traf.alt, self.traf.gs, self.traf.delalt, \
        self.traf.cas, self.vmto, self.vmic, self.vmap, self.vmcr, self.vmld, self.traf.bank, self.traf.bphase, \
        self.traf.hdgsel, self.traf.bada)

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
                   
        kf = (self.phase==1)*(self.etype==1)*coeffBS.d_kj[0] + \
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

        # print "KF climb",(self.phase == 3)*(self.etype==1)*(self.traf.delalt>1)*coeffBS.d_kj[1]
        # print "KF cruise",(self.phase == 3)*(self.etype==1)*((self.traf.delalt>-1 )& (self.traf.delalt<1))*coeffBS.d_kj[2] 
        # print "KF descent",(self.phase == 3)*(self.etype==1)*(self.traf.delalt<-1)*coeffBS.d_kj[3]

        # print CD0f, kf


        # line for kf-c and kf+fuel
        cd = self.CD0*CD0f + self.k*kf*(cl**2)

        # line for w/o
        #cd = self.CD0+self.k*(cl**2)
        
        # print "CL", cl, "CD", cd
        # compute drag: CD = CD0 + CDi * CL^2 and D = rho/2*VTAS^2*CD*S
        self.D = cd*self.qS

        # energy share factor and crossover altitude  
        epsalt = np.array([0.001]*self.traf.ntraf)   
        self.climb = np.array(self.traf.delalt > epsalt)
        self.descent = np.array(self.traf.delalt<epsalt)
  
        #crossover altitude (BADA User Manual 3.12, p. 12)
        self.atrans = (1000/6.5)*(T0*(1-((((1+gamma1*(self.refcas/a0)**2)**(gamma2))-1) /  \
        (((1+gamma1*self.refma**2)**(gamma2))-1))**((-(beta)*R)/g0)))

        # crossover altitiude
        self.traf.abco = np.array(self.traf.alt>self.atrans)
        self.traf.belco = np.array(self.traf.alt<self.atrans)

        # energy share factor
        self.ESF = esf(self.traf.abco, self.traf.belco, self.traf.alt, self.traf.M,\
                  self.climb, self.descent, self.traf.delspd)


        # THRUST
        # determine vertical speed
        swvs = (np.abs(self.traf.desvs) > self.traf.eps)
        vspd = swvs * self.traf.desvs + (1. - swvs) * self.traf.avsdef * np.sign(self.traf.delalt)
        swaltsel = np.abs(self.traf.delalt) > np.abs(2. * self.traf.perfdt * np.abs(vspd))
        self.traf.vs = swaltsel * vspd  

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
        #self.mass = self.mass - self.ff*self.traf.perfdt/60. # Use fuelflow in kg/min

        # print self.traf.id, self.phase, self.traf.alt/ft, self.traf.tas/kts, self.traf.cas/kts, self.traf.M,  \
        # self.Thr, self.D, self.ff,  cl, cd, self.traf.vs/fpm, self.ESF,self.atrans, self.maxthr, \
        # self.vmto/kts, self.vmic/kts ,self.vmcr/kts, self.vmap/kts, self.vmld/kts, \
        # CD0f, kf, self.hmaxact

        return



    def limits(self):
        """Flight envelope"""

        # combine minimum speeds and flight phases. Phases initial climb, cruise
        # and approach use the same CLmax and thus the same function for Vmin
        self.vmto = vtas2cas(self.vm_to*np.sqrt(self.mass/self.traf.rho), self.traf.alt)
        self.vmic = vtas2cas(np.sqrt(2*self.mass*g0/(self.traf.rho*self.clmaxcr*self.Sref)), self.traf.alt)
        self.vmcr = self.vmic
        self.vmap = self.vmic
        self.vmld = vtas2cas(self.vm_ld*np.sqrt(self.mass/self.traf.rho), self.traf.alt)

        # summarize
        # note: aircraft on ground may be pushed back
        self.vmin = (self.phase==1)*self.vmto + ((self.phase==2) + (self.phase==3) + (self.phase==4))*self.vmcr + \
                    (self.phase==5)*self.vmld + (self.phase==6)*-10.0

        # forwarding to tools
        self.traf.lspd, self.traf.lalt, self.traf.lvs, self.traf.ama = \
        limits(self.traf.desspd, self.traf.lspd, self.vmin, self.vmo, self.mmo,\
        self.traf.M, self.traf.ama, self.traf.alt, self.hmaxact, self.traf.desalt, self.traf.lalt,\
        self.maxthr, self.Thr,self.traf.lvs, self.D, self.traf.tas, self.mass, self.ESF)        

        return

    def engchange(self, acid, engid):
        """change of engines - for jet aircraft only!"""
        idx = self.traf.id.index(acid)

        self.jetengidx = coeffBS.jetenlist.index(coeffBS.engines[idx][engid])
  
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

        # Create datalog instance
        self.log = Datalog()
        return


    def reset(self):
        """RESET DATABASE"""

        # engine
        self.etype = np.array ([]) # jet, turboprop or piston

        # masses and dimensions
        self.mass = np.array([]) # effective mass [kg]
        # self.mref = np.array([]) # ref. mass [kg]: 70% between min and max. mass
        self.mmin = np.array([]) # OEW (or assumption) [kg]        
        self.mmax = np.array([]) # MTOW (or assumption) [kg]
        # self.mpyld = np.array([]) # MZFW-OEW (or assumption) [kg]
        self.gw = np.array([]) # weight gradient on max. alt [m/kg]
        self.Sref = np.array([]) # wing reference surface area [m^2]
    
        # flight enveloppe
        self.vmto = np.array([]) # min TO spd [m/s]
        self.vmic = np.array([]) # min climb spd [m/s]
        self.vmcr = np.array([]) # min cruise spd [m/s]
        self.vmap = np.array([]) # min approach spd [m/s]
        self.vmld = np.array([]) # min landing spd [m/s]   
        self.vmin = np.array([]) # min speed over all phases [m/s]   
    
        self.vmo =  np.array([]) # max operating speed [m/s]
        self.mmo =  np.array([]) # max operating mach number [-]
        self.hmax = np.array([]) # max. alt above standard MSL (ISA) at MTOW [m]
        self.hmaxact = np.array([]) # max. alt depending on temperature gradient [m]
        self.hmo =  np.array([]) # max. operating alt abov standard MSL [m]
        self.gt =   np.array([]) # temp. gradient on max. alt [ft/k]
        self.maxthr = np.array([]) # maximum thrust [N]
    
        # Buffet Coefficients
        self.clbo = np.array([]) # buffet onset lift coefficient [-]
        self.k =    np.array([]) # buffet coefficient [-]
        self.cm16 = np.array([]) # CM16
        
        # reference CAS speeds
        self.cascl  = np.array([]) # climb [m/s]
        self.cascr  = np.array([]) # cruise [m/s]
        self.casdes = np.array([]) # descent [m/s]
        self.refcas    = np.array([]) # nominal cruise CAS  [m/s]
        
        #reference mach numbers [-] 
        self.macl = np.array([]) # climb 
        self.macr = np.array([]) # cruise 
        self.mades = np.array([]) # descent 
        self.refma = np.array([]) # nominal cruise Mach at 35000 ft
        
        # parasitic drag coefficients per phase [-]
        self.cd0to = np.array([]) # phase takeoff 
        self.cd0ic = np.array([]) # phase initial climb
        self.cd0cr = np.array([]) # phase cruise
        self.cd0ap = np.array([]) # phase approach
        self.cd0ld = np.array([]) # phase land
        self.gear  = np.array([]) # drag due to gear down
        
        # induced drag coefficients per phase [-]
        self.cd2to = np.array([]) # phase takeoff
        self.cd2ic = np.array([]) # phase initial climb
        self.cd2cr = np.array([]) # phase cruise
        self.cd2ap = np.array([]) # phase approach
        self.cd2ld = np.array([]) # phase land
    
        # max climb thrust coefficients
        self.ctcth1 = np.array([]) # jet/piston [N], turboprop [ktN]
        self.ctcth2 = np.array([]) # [ft]
        self.ctcth3 = np.array([]) # jet [1/ft^2], turboprop [N], piston [ktN]
    
        # reduced climb power coefficient
        self.cred = np.array([]) # [-]
    
        # 1st and 2nd thrust temp coefficient 
        self.ctct1 = np.array([]) # [k]
        self.ctct2 = np.array([]) # [1/k]
        self.dtemp = np.array([]) # [k]
    
        # Descent Fuel Flow Coefficients
        # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
        self.ctdesl =  np.array([]) # low alt descent thrust coefficient [-]
        self.ctdesh =  np.array([]) # high alt descent thrust coefficient [-]
        self.ctdesa =  np.array([]) # approach thrust coefficient [-]
        self.ctdesld = np.array([]) # landing thrust coefficient [-]
    
        # transition altitude for calculation of descent thrust
        self.hpdes = np.array([]) # [m]
        
        # crossover altitude
        self.atrans = np.array([]) # [m]
        self.ESF = np.array([]) # [-]  
        
        # reference speed during descent
        self.vdes = np.array([]) # [m/s]
        self.mdes = np.array([]) # [-]
        
        # flight phase
        self.phase = np.array([])
        
        # Thrust specific fuel consumption coefficients
        self.cf1 = np.array([]) # jet [kg/(min*kN)], turboprop [kg/(min*kN*knot)], piston [kg/min]
        self.cf2 = np.array([]) # [knots]
        self.cf3 = np.array([]) # [kg/min]
        self.cf4 = np.array([]) # [ft]
        self.cf5 = np.array([]) # [-]

        # performance
        self.Thr = np.array([]) # thrust
        self.D   = np.array([]) # drag
        self.ff  = np.array([]) # fuel flow
    
        # ground
        self.tol = np.array([]) # take-off length[m]
        self.ldl = np.array([]) #landing length[m]
        self.ws  = np.array([]) # wingspan [m]
        self.len = np.array([]) # aircraft length[m] 

        return
       


    def create(self, actype):
        """CREATE NEW AIRCRAFT"""
        # note: coefficients are initialized in SI units

        # general        
        # designate aircraft to its aircraft type
        try:
            self.coeffidx = coeff.atype.index(actype)
        except:
            self.coeffidx = 0
            if not self.warned:
                  print "Aircraft is using default B747-400 performance."
            self.warned = True
        # designate aicraft to its aircraft type
        self.etype = np.append(self.etype, coeff.etype[self.coeffidx])        
       
        # Initial aircraft mass is currently reference mass. 
        # BADA 3.12 also supports masses between 1.2*mmin and mmax
        # self.mref = np.append(self.mref, coeff.mref[self.coeffidx]*1000)         
        self.mass = np.append(self.mass, coeff.mref[self.coeffidx]*1000)   
        self.mmin = np.append(self.mmin, coeff.mmin[self.coeffidx]*1000)
        self.mmax = np.append(self.mmax, coeff.mmax[self.coeffidx]*1000)
        # self.mpyld = np.append(self.mpyld, coeff.mpyld[self.coeffidx]*1000)
        self.gw = np.append(self.gw, coeff.gw[self.coeffidx]*ft)
        
        # Surface Area [m^2]
        self.Sref = np.append(self.Sref, coeff.Sref[self.coeffidx])

        # flight enveloppe
        # minimum speeds per phase
        self.vmto = np.append(self.vmto, coeff.vmto[self.coeffidx]*kts)
        self.vmic = np.append(self.vmic, coeff.vmic[self.coeffidx]*kts)
        self.vmcr = np.append(self.vmcr, coeff.vmcr[self.coeffidx]*kts)
        self.vmap = np.append(self.vmap, coeff.vmap[self.coeffidx]*kts)
        self.vmld = np.append(self.vmld, coeff.vmld[self.coeffidx]*kts)    
        self.vmin = np.append(self.vmin, 0.)
        self.vmo = np.append(self.vmo, coeff.vmo[self.coeffidx]*kts)
        self.mmo = np.append(self.mmo, coeff.mmo[self.coeffidx])
        
        # max. altitude parameters
        self.hmo = np.append(self.hmo, coeff.hmo[self.coeffidx]*ft)        
        self.hmax = np.append(self.hmax, coeff.hmax[self.coeffidx]*ft)
        self.hmaxact = np.append(self.hmaxact, coeff.hmax[self.coeffidx]*ft) # initialize with hmax
        self.gt = np.append(self.gt, coeff.gt[self.coeffidx]*ft)
        
        # max thrust setting
        self.maxthr = np.append(self.maxthr, 1000000.) # initialize with excessive setting to avoid unrealistic limit setting

        # Buffet Coefficients
        self.clbo = np.append(self.clbo, coeff.clbo[self.coeffidx])
        self.k = np.append(self.k, coeff.k[self.coeffidx])
        self.cm16 = np.append(self.cm16, coeff.cm16[self.coeffidx])

        # reference speeds
        # reference CAS speeds
        self.cascl = np.append(self.cascl, coeff.cascl[self.coeffidx]*kts)
        self.cascr = np.append(self.cascr, coeff.cascr[self.coeffidx]*kts)
        self.casdes = np.append(self.casdes, coeff.casdes[self.coeffidx]*kts)

        # reference mach numbers
        self.macl = np.append(self.macl, coeff.macl[self.coeffidx])
        self.macr = np.append(self.macr, coeff.macr[self.coeffidx] )
        self.mades = np.append(self.mades, coeff.mades[self.coeffidx] )      

        # reference speed during descent
        self.vdes = np.append(self.vdes, coeff.vdes[self.coeffidx]*kts)
        self.mdes = np.append(self.mdes, coeff.mdes[self.coeffidx])

        # aerodynamics                
        # parasitic drag coefficients per phase
        self.cd0to = np.append(self.cd0to, coeff.cd0to[self.coeffidx])
        self.cd0ic = np.append(self.cd0ic, coeff.cd0ic[self.coeffidx])
        self.cd0cr = np.append(self.cd0cr, coeff.cd0cr[self.coeffidx])
        self.cd0ap = np.append(self.cd0ap, coeff.cd0ap[self.coeffidx])
        self.cd0ld = np.append(self.cd0ld, coeff.cd0ld[self.coeffidx])
        self.gear = np.append(self.gear, coeff.gear[self.coeffidx])

        # induced drag coefficients per phase
        self.cd2to = np.append(self.cd2to, coeff.cd2to[self.coeffidx])
        self.cd2ic = np.append(self.cd2ic, coeff.cd2ic[self.coeffidx])
        self.cd2cr = np.append(self.cd2cr, coeff.cd2cr[self.coeffidx])
        self.cd2ap = np.append(self.cd2ap, coeff.cd2ap[self.coeffidx])
        self.cd2ld = np.append(self.cd2ld, coeff.cd2ld[self.coeffidx])

        # reduced climb coefficient
        #jet
        if self.etype [self.traf.ntraf-1] == 1:
            self.cred = np.append(self.cred, coeff.credj)
        # turboprop
        elif self.etype [self.traf.ntraf-1]  ==2:
            self.cred = np.append(self.cred, coeff.credt)
        #piston
        else:
            self.cred = np.append(self.cred, coeff.credp)

        # NOTE: model only validated for jet and turbo aircraft
        if not self.warned2 and self.etype[self.traf.ntraf-1] == 3:
            print "Using piston aircraft performance.",
            print "Not valid for real performance calculations."
            self.warned2 = True        

        # performance

        # max climb thrust coefficients
        self.ctcth1 = np.append(self.ctcth1, coeff.ctcth1[self.coeffidx]) # jet/piston [N], turboprop [ktN]
        self.ctcth2 = np.append(self.ctcth2, coeff.ctcth2[self.coeffidx]) # [ft]
        self.ctcth3 = np.append(self.ctcth3, coeff.ctcth3[self.coeffidx]) # jet [1/ft^2], turboprop [N], piston [ktN]

        # 1st and 2nd thrust temp coefficient 
        self.ctct1 = np.append(self.ctct1, coeff.ctct1[self.coeffidx]) # [k]
        self.ctct2 = np.append(self.ctct2, coeff.ctct2[self.coeffidx]) # [1/k]
        self.dtemp = np.append(self.dtemp, 0.) # [k], difference from current to ISA temperature. At the moment: 0, as ISA environment

        # Descent Fuel Flow Coefficients
        # Note: Ctdes,app and Ctdes,lnd assume a 3 degree descent gradient during app and lnd
        self.ctdesl = np.append(self.ctdesl, coeff.ctdesl[self.coeffidx])
        self.ctdesh = np.append(self.ctdesh, coeff.ctdesh[self.coeffidx])
        self.ctdesa = np.append(self.ctdesa, coeff.ctdesa[self.coeffidx])
        self.ctdesld = np.append(self.ctdesld, coeff.ctdesld[self.coeffidx])

        # transition altitude for calculation of descent thrust
        self.hpdes = np.append(self.hpdes, coeff.hpdes[self.coeffidx]*ft)
        self.ESF   = np.append(self.ESF, 1.) # neutral initialisation

        # flight phase
        self.phase = np.append(self.phase, 0)

         # Thrust specific fuel consumption coefficients
        self.cf1 = np.append(self.cf1, coeff.cf1[self.coeffidx])      
        # prevent from division per zero in fuelflow calculation
        if coeff.cf2[self.coeffidx]==0:
            self.cf2 = np.append(self.cf2, 1) 
        else:
            self.cf2 = np.append(self.cf2, coeff.cf2[self.coeffidx])
        self.cf3 = np.append(self.cf3, coeff.cf3[self.coeffidx])       
        # prevent from division per zero in fuelflow calculation
        if coeff.cf2[self.coeffidx]==0:
            self.cf4 = np.append(self.cf4, 1)
        else:
            self.cf4 = np.append(self.cf4, coeff.cf4[self.coeffidx])            
        self.cf5 = np.append(self.cf5, coeff.cf5[self.coeffidx])

        self.Thr = np.append(self.Thr, 0.)
        self.D = np.append(self.D, 0.)         
        self.ff = np.append(self.ff, 0.)

        # ground
        self.tol = np.append(self.tol, coeff.tol[self.coeffidx])
        self.ldl = np.append(self.ldl, coeff.ldl[self.coeffidx])
        self.ws = np.append(self.ws, coeff.ws[self.coeffidx])
        self.len = np.append(self.len, coeff.len[self.coeffidx])         
        return


    def delete(self, idx):
        """Delete REMOVED AIRCRAFT"""        
        
        # mass and dimensions
        # self.mref = np.delete(self.mref, idx)
        self.mass = np.delete(self.mass, idx)
        self.mmin = np.delete(self.mmin, idx)
        self.mmax = np.delete(self.mmax, idx)
        # self.mpyld = np.delete(self.mpyld, idx)
        self.gw = np.delete(self.gw, idx)

        self.Sref = np.delete(self.Sref, idx)

        # engine
        self.etype = np.delete(self.etype, idx)

        # flight enveloppe
        # speeds
        self.vmto = np.delete(self.vmto, idx)
        self.vmic = np.delete(self.vmic, idx)
        self.vmcr = np.delete(self.vmcr, idx)
        self.vmap = np.delete(self.vmap, idx)
        self.vmld = np.delete(self.vmld, idx) 
        self.vmin = np.delete(self.vmin, idx)
        self.vmo = np.delete(self.vmo, idx)
        self.mmo = np.delete(self.mmo, idx)

        # altitude
        self.hmo = np.delete(self.hmo, idx)
        self.hmax = np.delete(self.hmax, idx)
        self.hmaxact = np.delete (self.hmaxact, idx)
        self.maxthr = np.delete(self.maxthr, idx)
        self.gt = np.delete(self.gt, idx)

        # buffet coefficients
        self.clbo = np.delete(self.clbo, idx)
        self.k = np.delete(self.k, idx)
        self.cm16 = np.delete(self.cm16, idx)

        # reference speeds        
        # reference CAS
        self.cascl = np.delete(self.cascl, idx)
        self.cascr = np.delete(self.cascr, idx)
        self.casdes = np.delete(self.casdes, idx)
        self.refcas = np.delete(self.refcas, idx)

        # reference Mach
        self.macl = np.delete(self.macl, idx)
        self.macr = np.delete(self.macr, idx)
        self.mades = np.delete(self.mades, idx)      
        self.refma = np.delete(self.refma, idx)        

        # reference speed during descent
        self.vdes = np.delete(self.vdes, idx)
        self.mdes = np.delete(self.mdes, idx)

        # aerodynamics
        self.qS = np.delete(self.qS, idx)

        # parasitic drag coefficients per phase
        self.cd0to = np.delete(self.cd0to, idx)
        self.cd0ic = np.delete(self.cd0ic, idx)
        self.cd0cr = np.delete(self.cd0cr, idx)
        self.cd0ap = np.delete(self.cd0ap, idx)
        self.cd0ld = np.delete(self.cd0ld, idx)
        self.gear = np.delete(self.gear, idx)

        # induced drag coefficients per phase
        self.cd2to = np.delete(self.cd2to, idx)
        self.cd2ic = np.delete(self.cd2ic, idx)
        self.cd2cr = np.delete(self.cd2cr, idx)
        self.cd2ap = np.delete(self.cd2ap, idx)
        self.cd2ld = np.delete(self.cd2ld, idx)

        # performance        
        # reduced climb coefficient
        self.cred = np.delete(self.cred, idx)

        # max climb thrust coefficients
        self.ctcth1 = np.delete(self.ctcth1, idx) # jet/piston [N], turboprop [ktN]
        self.ctcth2 = np.delete(self.ctcth2, idx) # [ft]
        self.ctcth3 = np.delete(self.ctcth3, idx) # jet [1/ft^2], turboprop [N], piston [ktN]

        # 1st and 2nd thrust temp coefficient 
        self.ctct1 = np.delete(self.ctct1, idx) # [k]
        self.ctct2 = np.delete(self.ctct2, idx) # [1/k]
        self.dtemp = np.delete(self.dtemp, idx) # [k]

        # Descent Fuel Flow Coefficients
        self.ctdesl = np.delete(self.ctdesl, idx)
        self.ctdesh = np.delete(self.ctdesh, idx)
        self.ctdesa = np.delete(self.ctdesa, idx)
        self.ctdesld = np.delete(self.ctdesld, idx)

        # transition altitude for calculation of descent thrust
        self.hpdes = np.delete(self.hpdes, idx)
        self.hact = np.delete(self.hact, idx)

        # crossover altitude
        self.atrans = np.delete(self.atrans, idx)
        self.ESF = np.delete (self.ESF, idx)

        # flight phase
        self.phase = np.delete(self.phase, idx)
        self.bank = np.delete(self.bank, idx)

        # Thrust specific fuel consumption coefficients
        self.cf1 = np.delete(self.cf1, idx)
        self.cf2 = np.delete(self.cf2, idx)
        self.cf3 = np.delete(self.cf3, idx)
        self.cf4 = np.delete(self.cf4, idx)
        self.cf5 = np.delete(self.cf5, idx)

        self.Thr = np.delete(self.Thr, idx)
        self.D = np.delete(self.D, idx)         
        self.ff = np.delete(self.ff, idx)

        # ground
        self.tol = np.delete(self.tol, idx)
        self.ldl = np.delete(self.ldl, idx)
        self.ws  = np.delete(self.ws, idx)
        self.len = np.delete(self.len, idx)
        
        return


    def perf(self):
        """AIRCRAFT PERFORMANCE"""
        # flight phase
        self.phase, self.bank = \
        phases(self.traf.alt, self.traf.gs, self.traf.delalt, \
        self.traf.cas, self.vmto, self.vmic, self.vmap, self.vmcr, self.vmld, self.traf.bank, self.traf.bphase, \
        self.traf.hdgsel, self.traf.bada)

        # AERODYNAMICS
        # Lift
        self.qS = 0.5*self.traf.rho*np.maximum(1.,self.traf.tas)*np.maximum(1.,self.traf.tas)*self.Sref
        cl = self.mass*g0/(self.qS*np.cos(self.bank))*(self.phase!=6)+ 0.*(self.phase==6)

        # Drag
        # Drag Coefficient

        # phases TO, IC, CR
        cdph = self.cd0cr+self.cd2cr*(cl**2)

        # phase AP
        cdapp = self.cd0ap+self.cd2ap*(cl**2)

        # in case approach coefficients in OPF-Files are set to zero: 
        #Use cruise values instead
        cdapp = cdapp + (cdapp ==0)*cdph

        # phase LD
        cdld = self.cd0ld + self.gear + self.cd2ld*(cl**2)

        # in case landing coefficients in OPF-Files are set to zero: 
        #Use cruise values instead
        cdld = cdld + (cdld ==0)*cdph   

        # now combine phases            
        cd = (self.phase==1)*cdph + (self.phase==2)*cdph + (self.phase==3)*cdph \
            + (self.phase==4)*cdapp + (self.phase ==5)*cdld  

        # Drag:
        self.D = cd*self.qS 

        # energy share factor and crossover altitude  

        # conditions
        epsalt = np.array([0.001]*self.traf.ntraf)   
        self.climb = np.array(self.traf.delalt > epsalt)
        self.descent = np.array(self.traf.delalt<epsalt)
        lvl = np.array(np.abs(self.traf.delalt)<0.0001)*1
        
        # designate reference CAS to phases
        cascl = self.cascl*self.climb
        cascr = self.cascr*lvl
        casdes = self.casdes*self.descent
        
        self.refcas = np.maximum.reduce([cascl, cascr, casdes])

        # designate reference Ma to phases
        macl = self.macl*self.climb
        macr = self.macr*(np.abs(self.traf.delalt) < epsalt)
        mades = self.mades*self.descent

        self.refma = np.maximum.reduce([macl, macr, mades])

        # crossover altitude (BADA User Manual 3.12, p. 12)
        self.atrans = (1000/6.5)*(T0*(1-((((1+gamma1*(self.refcas/a0)**2)**(gamma2))-1) /  \
        (((1+gamma1*self.refma**2)**(gamma2))-1))**((-(beta)*R)/g0)))

        # crossover altitiude
        self.traf.abco = np.array(self.traf.alt>self.atrans)
        self.traf.belco = np.array(self.traf.alt<self.atrans)

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
        Tj = self.ctcth1* (1-(self.traf.alt/ft)/self.ctcth2+self.ctcth3*(self.traf.alt/ft)**2) 

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
            T = ((self.traf.avs!=0)*(((self.traf.desvs*self.mass*g0)/     \
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
        self.mass = self.mass - self.ff*self.traf.perfdt # Use fuelflow in kg/min
        return

    
    def limits(self):
        """FLIGHT ENVELPOE"""        
        # summarize minimum speeds
        self.vmin = vmin(self.vmto, self.vmic, self.vmcr, self.vmap, self.vmld, self.phase)        

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
        self.traf.lspd, self.traf.lalt, self.traf.lvs, self.traf.ama = \
        limits(self.traf.desspd, self.traf.lspd, self.vmin, self.vmo, self.mmo,\
        self.traf.M, self.traf.ama, self.traf.alt, self.hmaxact, self.traf.desalt, self.traf.lalt,\
        self.maxthr, self.Thr,self.traf.lvs,  self.D, self.traf.tas, self.mass, self.ESF)        
        
        return

        #------------------------------------------------------------------------------
        #DEBUGGING

        #record data 
        # self.log.write(self.traf.perfdt, str(self.traf.alt[0]), str(self.traf.tas[0]), str(self.D[0]), str(self.T[0]), str(self.ff[0]),  str(self.traf.vs[0]), str(cd[0]))
        # self.log.save()

        # print self.id, self.phase, self.alt/ft, self.tas/kts, self.cas/kts, self.M,  \
        # self.Thr, self.D, self.ff,  cl, cd, self.vs/fpm, self.ESF,self.atrans, maxthr, \
        # self.vmto/kts, self.vmic/kts ,self.vmcr/kts, self.vmap/kts, self.vmld/kts, \
        # CD0f, kf, self.hmaxact   


