import numpy as np

from math import *
from ..tools.aero import ft, g0, a0, T0, rho0, gamma1, gamma2,  beta, R
from ..tools.aero_np import vtas2cas

from ..tools.performance import esf, phases, limits

from params import CoeffBS

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







