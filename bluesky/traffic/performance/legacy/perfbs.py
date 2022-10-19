""" BlueSky aircraft performance calculations."""
from math import *
import numpy as np
import bluesky as bs
from bluesky.tools.aero import ft, g0, a0, T0, rho0, gamma1, gamma2,  beta, R, \
    kts, lbs, inch, sqft, fpm, vtas2cas, vcas2tas
from bluesky.traffic.performance.perfbase import PerfBase
from bluesky.traffic.performance.legacy.performance import esf, phases, calclimits, PHASE
from bluesky import settings

from bluesky.traffic.performance.legacy.coeff_bs import CoeffBS

# Register settings defaults
settings.set_variable_defaults(performance_dt=1.0, verbose=False)
coeffBS = CoeffBS()


class Legacy(PerfBase):
    def __init__(self):
        super().__init__()
        self.warned  = False    # Flag: Did we warn for default perf parameters yet?
        self.warned2 = False    # Flag: Use of piston engine aircraft?

        # prepare for coefficient readin
        coeffBS.coeff()

        with self.settrafarrays():
            # index of aircraft types in library
            self.coeffidxlist = np.array([])

            # geometry and weight
            self.mass         = np.array([]) # Mass [kg]
            self.Sref         = np.array([]) # Wing surface area [m^2]

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
            self.vmo          = np.array([]) # max CAS
            self.mmo          = np.array([]) # max Mach

            self.hmaxact      = np.array([]) # max. altitude
            self.maxthr       = np.array([]) # maximum thrust

            # aerodynamics
            self.CD0          = np.array([]) # parasite drag coefficient
            self.k            = np.array([]) # induced drag factor
            self.clmaxcr      = np.array([]) # max. cruise lift coefficient
            self.qS           = np.array([]) # Dynamic air pressure [Pa]

            # engines
            self.n_eng        = np.array([]) # Number of engines
            self.etype        = np.array([]) # jet /turboprop

            # jet engines:
            self.rated_thrust = np.array([]) # rated thrust (all engines)
            self.SFC = np.array([]) # specific fuel consumption in cruise
            self.fuelflow = np.array([]) # fuel flow
            self.ffto = np.array([]) # fuel flow takeoff
            self.ffcl = np.array([]) # fuel flow climb
            self.ffcr = np.array([]) # fuel flow cruise
            self.ffid = np.array([]) # fuel flow idle
            self.ffap = np.array([]) # fuel flow approach

            # turboprop engines
            self.P = np.array([]) # avaliable power at takeoff conditions
            self.PSFC_TO = np.array([]) # specific fuel consumption takeoff
            self.PSFC_CR = np.array([]) # specific fuel consumption cruise

            self.thrust = np.array([]) # Thrust
            self.thrust_pilot = np.array([])   # thrust required for pilot settings
            self.D = np.array([]) # Drag
            self.ESF = np.array([]) # Energy share factor according to EUROCONTROL

            # flight phase
            self.phase = np.array([]) # flight phase
            self.bank = np.array([]) # bank angle
            self.post_flight = np.array([]) # check for ground mode:
                                              #taxi prior of after flight
            self.pf_flag = np.array([])

            self.engines = []           # avaliable engine type per aircraft type

            # limit settings
            self.limspd      = np.array([])  # limit speed
            self.limspd_flag = np.array([], dtype=bool)  # flag for limit spd - we have to test for max and min
            self.limalt      = np.array([])  # limit altitude
            self.limalt_flag = np.array([])  # A need to limit altitude has been detected
            self.limvs       = np.array([])  # limit vertical speed due to thrust limitation
            self.limvs_flag  = np.array([])  # A need to limit V/S detected
        self.eta = 0.8          # propeller efficiency according to Raymer
        self.thrust_settings = np.array([1., 0.85, 0.07, 0.3 ]) # Thrust settings per flight phase according to ICAO

    def create(self, n=1):
        super().create(n)
        """CREATE NEW AIRCRAFT"""
        actypes = bs.traf.type[-n:]
        coeffidx = []

        for actype in actypes:
            if actype in coeffBS.atype:
                coeffidx.append(coeffBS.atype.index(actype))
            else:
                coeffidx.append(0)
                if not settings.verbose:
                    if not self.warned:
                        print("Aircraft is using default B747-400 performance.")
                        self.warned = True
                else:
                    print("Flight " + bs.traf.id[-1] + " has an unknown aircraft type, " + actype + ", BlueSky then uses default B747-400 performance.")
        coeffidx = np.array(coeffidx)

        # note: coefficients are initialized in SI units

        self.coeffidxlist[-n:]      = coeffidx
        self.mass[-n:]              = coeffBS.MTOW[coeffidx] # aircraft weight
        self.Sref[-n:]              = coeffBS.Sref[coeffidx] # wing surface reference area
        self.etype[-n:]             = coeffBS.etype[coeffidx] # engine type of current aircraft
        self.engines[-n:]           = [coeffBS.engines[c] for c in coeffidx]

        # speeds
        self.refma[-n:]             = coeffBS.cr_Ma[coeffidx] # nominal cruise Mach at 35000 ft
        self.refcas[-n:]            = vtas2cas(coeffBS.cr_spd[coeffidx], 35000*ft) # nominal cruise CAS
        self.gr_acc[-n:]            = coeffBS.gr_acc[coeffidx] # ground acceleration
        self.gr_dec[-n:]            = coeffBS.gr_dec[coeffidx] # ground acceleration

        # limits
        self.vm_to[-n:]             = coeffBS.vmto[coeffidx]
        self.vm_ld[-n:]             = coeffBS.vmld[coeffidx]
        self.mmo[-n:]               = coeffBS.max_Ma[coeffidx] # maximum Mach
        self.vmo[-n:]               = coeffBS.max_spd[coeffidx] # maximum CAS
        self.hmaxact[-n:]           = coeffBS.max_alt[coeffidx] # maximum altitude
        # self.vmto/vmic/vmcr/vmap/vmld/vmin are initialised as 0 by super.create

        # aerodynamics
        self.CD0[-n:]               = coeffBS.CD0[coeffidx]  # parasite drag coefficient
        self.k[-n:]                 = coeffBS.k[coeffidx]    # induced drag factor
        self.clmaxcr[-n:]           = coeffBS.clmax_cr[coeffidx]   # max. cruise lift coefficient
        self.ESF[-n:]               = 1.
        # self.D/qS are initialised as 0 by super.create

        # flight phase
        self.pf_flag[-n:]           = 1
        # self.phase/bank/post_flight are initialised as 0 by super.create

        # engines
        self.n_eng[-n:]              = coeffBS.n_eng[coeffidx] # Number of engines
        turboprops = self.etype[-n:] == 2

        propidx = []
        jetidx  = []

        for engine in self.engines[-n:]:
            # engine[0]: default to first engine in engine list for this aircraft
            if engine[0] in coeffBS.propenlist:
                propidx.append(coeffBS.propenlist.index(engine[0]))
            else:
                propidx.append(0)
            if engine[0] in coeffBS.jetenlist:
                jetidx.append(coeffBS.jetenlist.index(engine[0]))
            else:
                jetidx.append(0)

        propidx=np.array(propidx)
        jetidx =np.array(jetidx)
        # Make two index lists of the engine type, assuming jet and prop. In the end, choose which one to use

        self.P[-n:]         = np.where(turboprops, coeffBS.P[propidx]*self.n_eng[-n:]      , 1.)
        self.PSFC_TO[-n:]   = np.where(turboprops, coeffBS.PSFC_TO[propidx]*self.n_eng[-n:], 1.)
        self.PSFC_CR[-n:]   = np.where(turboprops, coeffBS.PSFC_CR[propidx]*self.n_eng[-n:], 1.)

        self.rated_thrust[-n:]      = np.where(turboprops, 1. , coeffBS.rated_thrust[jetidx]*coeffBS.n_eng[coeffidx])  # rated thrust (all engines)
        self.thrust[-n:]       = np.where(turboprops, 1. , coeffBS.rated_thrust[jetidx]*coeffBS.n_eng[coeffidx])  # initialize thrust with rated thrust
        self.thrust_pilot[-n:] = np.where(turboprops, 1. , coeffBS.rated_thrust[jetidx]*coeffBS.n_eng[coeffidx])  # initialize thrust with rated thrust
        self.maxthr[-n:]    = np.where(turboprops, 1. , coeffBS.rated_thrust[jetidx]*coeffBS.n_eng[coeffidx]*1.2)  # maximum thrust - initialize with 1.2*rThr
        self.SFC[-n:]       = np.where(turboprops, 1. , coeffBS.SFC[jetidx] )
        self.ffto[-n:]      = np.where(turboprops, 1. , coeffBS.ffto[jetidx]*coeffBS.n_eng[coeffidx]) / 60.0
        self.ffcl[-n:]      = np.where(turboprops, 1. , coeffBS.ffcl[jetidx]*coeffBS.n_eng[coeffidx]) / 60.0
        self.ffcr[-n:]      = np.where(turboprops, 1. , coeffBS.ffcr[jetidx]*coeffBS.n_eng[coeffidx]) / 60.0
        self.ffid[-n:]      = np.where(turboprops, 1. , coeffBS.ffid[jetidx]*coeffBS.n_eng[coeffidx]) / 60.0
        self.ffap[-n:]      = np.where(turboprops, 1. , coeffBS.ffap[jetidx]*coeffBS.n_eng[coeffidx]) / 60.0

    def update(self, dt):
        ''' Periodic update function for performance calculations. '''
        swbada = False # no-bada version
        delalt = bs.traf.selalt - bs.traf.alt
        # allocate aircraft to their flight phase
        self.phase, self.bank = phases(bs.traf.alt, bs.traf.gs, delalt,
            bs.traf.cas, self.vmto, self.vmic, self.vmap, self.vmcr, self.vmld,
            bs.traf.ap.bankdef, bs.traf.bphase, bs.traf.swhdgsel,swbada)

        # AERODYNAMICS
        # compute CL: CL = 2*m*g/(VTAS^2*rho*S)
        self.qS = 0.5*bs.traf.rho*np.maximum(1.,bs.traf.tas)*np.maximum(1.,bs.traf.tas)*self.Sref

        cl = self.mass*g0/(self.qS*np.cos(self.bank))*(self.phase!=6)+ 0.*(self.phase==6)

        # scaling factors for CD0 and CDi during flight phases according to FAA (2005): SAGE, V. 1.5, Technical Manual

        # For takeoff (phase = 6) drag is assumed equal to the takeoff phase
        CD0f = (self.phase==1)*(self.etype==1)*coeffBS.d_CD0j[0] + \
               (self.phase==2)*(self.etype==1)*coeffBS.d_CD0j[1]  + \
               (self.phase==3)*(self.etype==1)*coeffBS.d_CD0j[2] + \
               (self.phase==4)*(self.etype==1)*coeffBS.d_CD0j[3] + \
               (self.phase==5)*(self.etype==1)*(bs.traf.alt>=450.0)*coeffBS.d_CD0j[4] + \
               (self.phase==5)*(self.etype==1)*(bs.traf.alt<450.0)*coeffBS.d_CD0j[5] + \
               (self.phase==6)*(self.etype==1)*coeffBS.d_CD0j[0] + \
               (self.phase==1)*(self.etype==2)*coeffBS.d_CD0t[0] + \
               (self.phase==2)*(self.etype==2)*coeffBS.d_CD0t[1]  + \
               (self.phase==3)*(self.etype==2)*coeffBS.d_CD0t[2] + \
               (self.phase==4)*(self.etype==2)*coeffBS.d_CD0t[3]
                   # (self.phase==5)*(self.etype==2)*(self.alt>=450)*coeffBS.d_CD0t[4] + \
                   # (self.phase==5)*(self.etype==2)*(self.alt<450)*coeffBS.d_CD0t[5]

        # For takeoff (phase = 6) induced drag is assumed equal to the takeoff phase
        kf =   (self.phase==1)*(self.etype==1)*coeffBS.d_kj[0] + \
               (self.phase==2)*(self.etype==1)*coeffBS.d_kj[1]  + \
               (self.phase==3)*(self.etype==1)*coeffBS.d_kj[2] + \
               (self.phase==4)*(self.etype==1)*coeffBS.d_kj[3] + \
               (self.phase==5)*(self.etype==1)*(bs.traf.alt>=450)*coeffBS.d_kj[4] + \
               (self.phase==5)*(self.etype==1)*(bs.traf.alt<450)*coeffBS.d_kj[5] + \
               (self.phase==6)*(self.etype==1)*coeffBS.d_kj[0] + \
               (self.phase==1)*(self.etype==2)*coeffBS.d_kt[0] + \
               (self.phase==2)*(self.etype==2)*coeffBS.d_kt[1]  + \
               (self.phase==3)*(self.etype==2)*coeffBS.d_kt[2] + \
               (self.phase==4)*(self.etype==2)*coeffBS.d_kt[3] + \
               (self.phase==5)*(self.etype==2)*(bs.traf.alt>=450)*coeffBS.d_kt[4] + \
               (self.phase==5)*(self.etype==2)*(bs.traf.alt<450)*coeffBS.d_kt[5]


        # drag coefficient
        cd = self.CD0*CD0f + self.k*kf*(cl*cl)

        # compute drag: CD = CD0 + CDi * CL^2 and D = rho/2*VTAS^2*CD*S
        self.D = cd*self.qS
        # energy share factor and crossover altitude
        epsalt = np.array([0.001]*bs.traf.ntraf)
        climb = np.array(delalt > epsalt)
        descent = np.array(delalt< -epsalt)

        # energy share factor
        delspd = bs.traf.aporasas.tas - bs.traf.tas
        selmach = bs.traf.selspd < 2.0
        self.ESF = esf(bs.traf.alt, bs.traf.M, climb, descent, delspd, selmach)

        # determine thrust
        self.thrust = (((bs.traf.vs*self.mass*g0)/(self.ESF*np.maximum(bs.traf.eps, bs.traf.tas))) + self.D)
        # determine thrust required to fulfill requests from pilot
        # self.thrust_pilot = (((bs.traf.aporasas.vs*self.mass*g0)/(self.ESF*np.maximum(bs.traf.eps, bs.traf.aporasas.tas))) + self.D)
        self.thrust_pilot = (((bs.traf.ap.vs*self.mass*g0)/(self.ESF*np.maximum(bs.traf.eps, bs.traf.aporasas.tas))) + self.D)

        # maximum thrust jet (Bruenig et al., p. 66):
        mt_jet = self.rated_thrust*(bs.traf.rho/rho0)**0.75

        # maximum thrust prop (Raymer, p.36):
        mt_prop = self.P*self.eta/np.maximum(bs.traf.eps, bs.traf.tas)

        # merge
        self.maxthr = mt_jet*(self.etype==1) + mt_prop*(self.etype==2)

        # Fuel Flow

        # jet aircraft
        # ratio current thrust/rated thrust
        pThr = self.thrust/self.rated_thrust
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
        PSFC = (((self.PSFC_CR - self.PSFC_TO) / 20000.0)*bs.traf.alt + self.PSFC_TO)*(bs.traf.alt<20.000) + \
                self.PSFC_CR*(bs.traf.alt >= 20.000)

        TSFC = PSFC*bs.traf.tas/(550.0*self.eta)

        # formula p.36 Raymer is missing here!
        ff_prop = self.thrust*TSFC*(self.etype==2)


        # combine
        self.fuelflow = np.maximum(0.0, ff_jet + ff_prop)

        # update mass
        self.mass -= self.fuelflow * dt

        # print bs.traf.id, self.phase, bs.traf.alt/ft, bs.traf.tas/kts, bs.traf.cas/kts, bs.traf.M,  \
        # self.thrust, self.D, self.ff,  cl, cd, bs.traf.vs/fpm, self.ESF,self.atrans, self.maxthr, \
        # self.vmto/kts, self.vmic/kts ,self.vmcr/kts, self.vmap/kts, self.vmld/kts, \
        # CD0f, kf, self.hmaxact


        # for aircraft on the runway and taxiways we need to know, whether they
        # are prior or after their flight
        self.post_flight = np.where(descent, True, self.post_flight)

        # when landing, we would like to stop the aircraft.
        bs.traf.aporasas.tas = np.where((bs.traf.alt <0.5)*(self.post_flight)*self.pf_flag, 0.0, bs.traf.aporasas.tas)
        # the impulse for reducing the speed to 0 should only be given once,
        # otherwise taxiing will be impossible afterwards
        self.pf_flag = np.where ((bs.traf.alt <0.5)*(self.post_flight), False, self.pf_flag)

        # define acceleration: aircraft taxiing and taking off use ground acceleration,
        # landing aircraft use ground deceleration, others use standard acceleration
        self.axmax = ((self.phase == PHASE['IC']) + (self.phase == PHASE['CR']) + (self.phase == PHASE['AP']) + (self.phase == PHASE['LD'])) * 0.5 \
            + ((self.phase == PHASE['TO']) + (self.phase == PHASE['GD'])*(1-self.post_flight)) * self.gr_acc  \
            + (self.phase == PHASE['GD']) * self.post_flight * self.gr_dec

    def limits(self, intent_v, intent_vs, intent_h, ax):
        """Flight envelope""" # Connect this with function limits in performance.py

        # combine minimum speeds and flight phases. Phases initial climb, cruise
        # and approach use the same CLmax and thus the same function for Vmin
        self.vmto = self.vm_to*np.sqrt(self.mass/bs.traf.rho)
        self.vmic = np.sqrt(2*self.mass*g0/(bs.traf.rho*self.clmaxcr*self.Sref))
        self.vmcr = self.vmic
        self.vmap = self.vmic
        self.vmld = self.vm_ld*np.sqrt(self.mass/bs.traf.rho)

        # summarize and convert to cas
        # note: aircraft on ground may be pushed back
        self.vmin = (self.phase==1)*vtas2cas(self.vmto, bs.traf.alt) + \
                        ((self.phase==2) + (self.phase==3) + (self.phase==4))*vtas2cas(self.vmcr, bs.traf.alt) + \
                            (self.phase==5)*vtas2cas(self.vmld, bs.traf.alt) + (self.phase==6)*-10.0


        # forwarding to tools
        self.limspd, self.limspd_flag, self.limalt, \
            self.limalt_flag, self.limvs, self.limvs_flag  =  calclimits(
                vtas2cas(intent_v, bs.traf.alt), bs.traf.gs, \
                self.vmto, self.vmin, self.vmo, self.mmo, \
                bs.traf.M, bs.traf.alt, self.hmaxact, \
                intent_h, intent_vs, self.maxthr, \
                self.thrust_pilot, self.D, bs.traf.tas, \
                self.mass, self.ESF, self.phase)

        # Update desired sates with values within the flight envelope
        # When CAS is limited, it needs to be converted to TAS as only this TAS is used later on!
        allowed_tas = np.where(self.limspd_flag, vcas2tas(self.limspd, bs.traf.alt), intent_v)

        # Autopilot selected altitude [m]
        allowed_alt = np.where(self.limalt_flag, self.limalt, intent_h)

        # Autopilot selected vertical speed (V/S)
        allowed_vs = np.where(self.limvs_flag, self.limvs, intent_vs)

        return allowed_tas, allowed_vs, allowed_alt

    def engchange(self, idx, engid=None):
        """change of engines - for jet aircraft only!"""
        if not engid:
            disptxt = "available engine types:\n" + '\n'.join(self.engines[idx]) + \
                      "\nChange engine with ENG acid engine_id"
            return False, disptxt
        engidx = self.engines[idx].index(engid)
        self.jetengidx = coeffBS.jetenlist.index(coeffBS.engines[idx][engidx])

        # exchange engine parameters

        self.rated_thrust[idx]   = coeffBS.rated_thrust[self.jetengidx]*coeffBS.n_eng[idx] # rated thrust (all engines)
        self.thrust[idx]    = coeffBS.rated_thrust[self.jetengidx]*coeffBS.n_eng[idx] # initialize thrust with rated thrust
        self.maxthr[idx] = coeffBS.rated_thrust[self.jetengidx]*coeffBS.n_eng[idx] # maximum thrust - initialize with 1.2*rThr
        self.SFC[idx]    = coeffBS.SFC[self.jetengidx]
        self.fuelflow[idx] = 0. # neutral initialisation
        self.ffto[idx]   = coeffBS.ffto[self.jetengidx]*coeffBS.n_eng[idx] / 60.0
        self.ffcl[idx]   = coeffBS.ffcl[self.jetengidx]*coeffBS.n_eng[idx] / 60.0
        self.ffcr[idx]   = coeffBS.ffcr[self.jetengidx]*coeffBS.n_eng[idx] / 60.0
        self.ffid[idx]   = coeffBS.ffid[self.jetengidx]*coeffBS.n_eng[idx] / 60.0
        self.ffap[idx]   = coeffBS.ffap[self.jetengidx]*coeffBS.n_eng[idx] / 60.0
        return
