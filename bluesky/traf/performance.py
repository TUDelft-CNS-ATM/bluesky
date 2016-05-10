"""
Aircraft Performance Modeling tools

Modules:
     esf (performance parameters): Energy Share Factor for climb/descent performance
     phases (performance parameters): define flight phases
     vmin (min speeds, phase)       : summarize minimum speeds
     limits (performance parameters): ensure aircraft remains within flight envelope

Created by  : Isabel Metz
Date        : February 2015

Modification  :
By            :
Date          :

"""

import numpy as np
from ..tools.aero import kts, ft, gamma, gamma1, gamma2, R, beta, g0, \
    vmach2cas, vcas2mach


#------------------------------------------------------------------------------
#
# FLIGHT PHASES
# based on the BADA 3.12 User Manual, chapter 3.5, p. 19, adapted
#------------------------------------------------------------------------------


def phases(alt, gs, delalt, cas, vmto, vmic, vmap,
           vmcr, vmld, bank, bphase, hdgsel, bada):

    # flight phases: TO (1), IC (2), CR (3), AP(4), LD(5), GD (6)
    #--> no holding phase yet
    #-------------------------------------------------
    # phase TO[1]: alt<400 and vs>0
    Talt = np.array(alt < (400. * ft))
    Tspd = np.array(gs > (30. * kts))
    Tvs  = np.array(delalt >= 0.) * 1.0
    to   = np.logical_and.reduce([Tspd, Talt, Tvs]) * 1

    #-------------------------------------------------
    # phase IC[2]: 400<alt<2000, vs>0
    Ialt = np.array((alt >= (400. * ft)) & (alt < (2000. * ft)))
    Ivs  = np.array(delalt > 0.)
    ic   = np.logical_and.reduce([Ialt, Ivs]) * 2

    #-------------------------------------------------
    #phase CR[3]: in climb above 2000ft, in descent
    # above 8000ft and below 8000ft if V>=Vmincr + 10kts

    # a. climb
    Caalt = np.array(alt >= (2000. * ft))
    Cavs  = np.array(delalt >= 0.)
    cra   = np.logical_and.reduce([Caalt, Cavs]) * 1

    #b. above 8000ft
    crb   = np.array(alt > (8000. * ft)) * 1

    #c. descent
    Ccalt = np.array(alt <= (8000. * ft))
    Ccvs  = np.array(delalt <= 0.)
    Ccspd = np.array(cas >= (vmcr + 10. * kts))
    crc   = np.logical_and.reduce([Ccalt, Ccvs, Ccspd]) * 1

    # merge climb and descent phase
    cr = np.maximum.reduce([cra, crb, crc]) * 3

    #-------------------------------------------------
    # phase AP[4]

    #a. alt<8000, Speed between Vmcr+10 and Vmapp+10, v<>0
    #altitude check
    Aaalt = np.array((alt > ft) & (alt <= (8000. * ft)))
    Aaspd = np.array(cas < (vmcr + 10. * kts))
    Aavs  = np.array(delalt <= 0.)
    apa   = np.logical_and.reduce([Aaalt, Aaspd , Aavs]) * 1
    #b. alt<3000ft, Vmcr+10>V>Vmap+10
    Abalt = np.array((alt > ft) & (alt <= (3000. * ft)))
    if bada:
        Abspd = np.array((cas >= (vmap + 10. * kts)) & (cas < (vmcr + 10. * kts)))
    else:
        Abspd = np.array(cas >= (vmap + 10. * kts))
    Abvs = np.array(delalt <= 0.)
    apb  = np.logical_and.reduce([Abalt, Abspd , Abvs]) * 1

    # merge a. and b.
    ap   = np.maximum.reduce([apa, apb]) * 4

    #-------------------------------------------------
    # phase LD[5]: alt<3000, Speed between Vmcr+10 and Vmap+10, vs<0
    # at the moment: 1000 for validation purposes
    Lalt = np.array(alt <= (1000 * ft))
    if bada:
        Lspd = np.array((cas < (vmap + 10.0 * kts)) & (gs >= (30.0 * kts)))
    else:
        Lspd = np.array(gs >= (30.0 * kts))
    Lvs = np.array(delalt <= 0.0)
    ld  = np.logical_and.reduce([Lalt, Lspd, Lvs]) * 5

    #-------------------------------------------------
    # phase GND: alt < 0.001ft
    gd = np.array((alt <= ft) & ((gs <= 30.0 * kts))) * 6

    #-------------------------------------------------
    # combine all phases
    phase = np.maximum.reduce([to, ic, ap, ld, cr, gd])

    to2 = np.where(phase == 1)
    ic2 = np.where(phase == 2)
    cr2 = np.where(phase == 3)
    ap2 = np.where(phase == 4)
    ld2 = np.where(phase == 5)
    gd2 = np.where(phase == 6)

    # assign aircraft to their nominal bank angle per phase
    bank[to2] = bphase[0]
    bank[ic2] = bphase[1]
    bank[cr2] = bphase[2]
    bank[ap2] = bphase[3]
    bank[ld2] = bphase[4]
    # to be refined! find value that comes closest to 1 if tan or cos
    bank[gd2] = bphase[5]

    # not turning aircraft do not have a bank angle.
    #hdgsel == True: Aircraft is turning
    noturn = np.array(hdgsel) * 100.0
    bank   = np.minimum(noturn, bank)

    return (phase, bank)


#------------------------------------------------------------------------------
#
# ENERGY SHARE FACTOR
# (BADA User Manual 3.12, p.15)
#
#-----------------------------------------------------------------------------
def esf(abco, belco, alt, M, climb, descent, delspd):

    # avoid wrong allocation due to infinitissimal speed changes
    cspd  = np.array((delspd <= 0.4) & (delspd >= -0.4))
    # accelerating or decelerating
    acc   = np.array(delspd > 0.4)
    dec   = np.array(delspd < -0.4)

    # tropopause
    abtp  = np.array(alt > 11000.0)
    beltp = np.array(alt < 11000.0)

    # constant Mach/CAS
    # case a: constant MA above TP
    efa   = np.logical_and.reduce([cspd, abco, abtp]) * 1

    # case b: constant MA below TP (at the moment just ISA: tISA = 1)
    # tISA = (self.temp-self.dtemp)/self.temp
    efb   = 1.0 / ((1.0 + ((gamma * R * beta) / (2.0 * g0)) * M**2)) \
        * np.logical_and.reduce([cspd, abco, beltp]) * 1

    # case c: constant CAS below TP (at the moment just ISA: tISA = 1)
    efc = 1.0 / (1.0 + (((gamma * R * beta) / (2.0 * g0)) * (M**2)) +
        ((1.0 + gamma1 * (M**2))**(-1.0 / (gamma - 1.0))) *
        (((1.0 + gamma1 * (M**2))**gamma2) - 1)) * \
        np.logical_and.reduce([cspd, belco, beltp]) * 1

    #case d: constant CAS above TP
    efd = 1.0 / (1.0 + ((1.0 + gamma1 * (M**2))**(-1.0 / (gamma - 1.0))) *
        (((1.0 + gamma1 * (M**2))**gamma2) - 1.0)) * \
        np.logical_and.reduce([cspd, belco, abtp]) * 1

    #case e: acceleration in climb
    clac   = np.logical_and.reduce([acc, climb]) * 1

    efe    = 0.3 * clac
    #case f: deceleration in descent
    decdes = np.logical_and.reduce([dec, descent]) * 1
    eff    = 0.3 * decdes
    #case g: deceleration in climb
    decl   = np.logical_and.reduce([dec, climb]) * 1
    efg    = 1.7 * decl
    #case h: acceleration in descent
    acdes  = np.logical_and.reduce([acc, descent]) * 1
    efh    = 1.7 * acdes

    # combine cases
    esf = np.maximum.reduce([efa, efb, efc, efd, efe, eff, efg, efh])

    # ESF of non-climbing/descending aircraft is zero what
    # leads to an error. Therefore, ESF for non-climbing aircraft is 1
    ESF = np.maximum(esf, np.array(esf == 0) * 1)

    return ESF


#------------------------------------------------------------------------------
#
# MINIMUM SPEEDS
#
#------------------------------------------------------------------------------
def vmin(vmto, vmic, vmcr, vmap, vmld, phase):
    vmin = (phase == 1) * vmto + (phase == 2) * vmic + (phase == 3) * vmcr + \
        (phase == 4) * vmap + (phase == 5) * vmld + (phase == 6) * 0.
    return vmin


#------------------------------------------------------------------------------
#
# LIMITS
#
#------------------------------------------------------------------------------
def limits(desspd, lspd, vmin, vmo, mmo, M, ama, alt, hmaxact, desalt,
           lalt, maxthr, Thr, lvs, D, tas, mass, ESF):
    # minimum speed
    vmincomp = np.less(desspd, vmin) * 1
    if (vmincomp.any() == 1):
        lspd = (vmincomp == 1) * (vmin + 1.0) + (vmincomp == 0) * 0.0
        # limit for above crossover
        ama = (vmincomp == 1) * vcas2mach(lspd, alt) + (vmincomp == 0) * ama
        #print "below minimum speed", lspd, ama

    # maximum speed

    # CAS
    vcomp = np.greater(vmo, desspd) * 1

    # set speed to max. possible speed if CAS>VMO
    if (vcomp.all() == 0):
        lspd = (vcomp == 0) * (vmo - 1.0) + (vcomp != 0) * 0.0
        # limit for above crossover
        ama = (vcomp == 0) * vcas2mach(lspd, alt) + (vcomp != 0) * ama
        # print "above max CAS speed", lspd

    # Mach
    macomp = np.greater(mmo, M) * 1
    # set speed to max. possible speed if Mach>MMO
    if (macomp.all() == 0):
        lspd = (macomp == 0) * vmach2cas((mmo - 0.001), alt) + (macomp != 0) * 0.0
        # limit for above crossover
        ama = (macomp == 0) * vcas2mach(lspd, alt) + (macomp != 0) * ama
        # print "above max Mach speed", lspd

    # remove non-needed speed limit
    ls = np.array(desspd == lspd)
    if (ls.any() == 1):
        lspd = (ls == 1) * 0.0

    # maximum altitude
    hcomp = np.greater(hmaxact, desalt) * 1

    # set altitude to max. possible altitude if alt>Hmax
    if(hcomp.all() == 0):
        lalt = (hcomp == 0) * (hmaxact - 1.0) + (hcomp != 0) * 0.0
        #  print "above max alt"

    # remove non-needed altitude limit
    la = np.array(desalt == lalt)
    if (la.any() == 1):
        lalt = (la == 1) * 0.0

    # thrust
    thrcomp = np.greater(maxthr, Thr) * 1
    if (thrcomp.all() == 0):
        Thr = (thrcomp == 0) * (maxthr - 1.0) + (thrcomp != 0) * Thr
        lvs = (thrcomp == 0) * (((Thr - D) * tas) / (mass * g0)) * ESF + \
            (thrcomp == 0) * 0.0

    return lspd, lalt, lvs, ama
