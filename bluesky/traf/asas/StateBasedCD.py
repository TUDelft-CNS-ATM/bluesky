"""
State-based conflict detection


"""
import numpy as np
from ...tools import geo
from ...tools.aero import nm


def detect(dbconf, traf, simt):
    if not dbconf.swasas:
        return

    # Reset lists before new CD
    dbconf.iconf        = [[] for ac in range(traf.ntraf)]
    dbconf.nconf        = 0
    dbconf.confpairs    = []
    dbconf.latowncpa    = []
    dbconf.lonowncpa    = []
    dbconf.altowncpa    = []

    dbconf.LOSlist_now  = []
    dbconf.conflist_now = []

    # Horizontal conflict ---------------------------------------------------------

    # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
    qdlst = geo.qdrdist_matrix(np.mat(traf.lat), np.mat(traf.lon),
                               np.mat(traf.adsblat), np.mat(traf.adsblon))

    # Convert results from mat-> array
    dbconf.qdr  = np.array(qdlst[0])  # degrees
    I           = np.eye(traf.ntraf)  # Identity matric of order ntraf
    dbconf.dist = np.array(qdlst[1]) * nm + 1e9 * I  # meters i to j

    # Transmission noise
    if traf.ADSBtransnoise:
        # error in the determined bearing between two a/c
        bearingerror = np.random.normal(0, traf.transerror[0], dbconf.qdr.shape)  # degrees
        dbconf.qdr += bearingerror
        # error in the perceived distance between two a/c
        disterror = np.random.normal(0, traf.transerror[1], dbconf.dist.shape)  # meters
        dbconf.dist += disterror

    # Calculate horizontal closest point of approach (CPA)
    qdrrad    = np.radians(dbconf.qdr)
    dbconf.dx = dbconf.dist * np.sin(qdrrad)  # is pos j rel to i
    dbconf.dy = dbconf.dist * np.cos(qdrrad)  # is pos j rel to i

    trkrad   = np.radians(traf.trk)
    dbconf.u = traf.gs * np.sin(trkrad).reshape((1, len(trkrad)))  # m/s
    dbconf.v = traf.gs * np.cos(trkrad).reshape((1, len(trkrad)))  # m/s

    # parameters received through ADSB
    adsbtrkrad = np.radians(traf.adsbtrk)
    adsbu = traf.adsbgs * np.sin(adsbtrkrad).reshape((1, len(adsbtrkrad)))  # m/s
    adsbv = traf.adsbgs * np.cos(adsbtrkrad).reshape((1, len(adsbtrkrad)))  # m/s

    du = dbconf.u - adsbu.T  # Speed du[i,j] is perceived eastern speed of i to j
    dv = dbconf.v - adsbv.T  # Speed dv[i,j] is perceived northern speed of i to j

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value

    vrel = np.sqrt(dv2)

    dbconf.tcpa = -(du * dbconf.dx + dv * dbconf.dy) / dv2 + 1e9 * I

    # Calculate CPA positions
    # xcpa = dbconf.tcpa * du
    # ycpa = dbconf.tcpa * dv

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = dbconf.dist * dbconf.dist - dbconf.tcpa * dbconf.tcpa * dv2

    # Check for horizontal conflict
    R2 = dbconf.R * dbconf.R
    swhorconf = dcpa2 < R2  # conflict or not

    # Calculate times of entering and leaving horizontal conflict
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
    dtinhor = dxinhor / vrel

    tinhor = np.where(swhorconf, dbconf.tcpa - dtinhor, 1e8)  # Set very large if no conf

    touthor = np.where(swhorconf, dbconf.tcpa + dtinhor, -1e8)  # set very large if no conf
    # swhorconf = swhorconf*(touthor>0)*(tinhor<dbconf.dtlook)

    # Vertical conflict -----------------------------------------------------------

    # Vertical crossing of disk (-dh,+dh)
    alt = traf.alt.reshape((1, traf.ntraf))
    adsbalt = traf.adsbalt.reshape((1, traf.ntraf))
    if traf.ADSBtransnoise:
        # error in the determined altitude of other a/c
        alterror = np.random.normal(0, traf.transerror[2], adsbalt.shape)  # degrees
        adsbalt += alterror

    dbconf.dalt = alt - adsbalt.T

    vs = traf.vs
    vs = vs.reshape(1, len(vs))

    avs = traf.adsbvs
    avs = avs.reshape(1, len(avs))

    dvs = vs - avs.T

    # Check for passing through each others zone
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero
    tcrosshi = (dbconf.dalt + dbconf.dh) / -dvs
    tcrosslo = (dbconf.dalt - dbconf.dh) / -dvs

    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)

    # Combine vertical and horizontal conflict-------------------------------------
    dbconf.tinconf = np.maximum(tinver, tinhor)

    dbconf.toutconf = np.minimum(toutver, touthor)

    swconfl = swhorconf * (dbconf.tinconf <= dbconf.toutconf) * \
        (dbconf.toutconf > 0.) * (dbconf.tinconf < dbconf.dtlookahead) \
        * (1. - I)

    # ----------------------------------------------------------------------
    # Update conflict lists
    # ----------------------------------------------------------------------
    if len(swconfl) == 0:
        return
    # Calculate CPA positions of traffic in lat/lon?

    # Select conflicting pairs: each a/c gets their own record
    confidxs            = np.where(swconfl)
    iown                = confidxs[0]
    ioth                = confidxs[1]

    # Store result
    dbconf.nconf        = len(confidxs[0])

    for idx in range(dbconf.nconf):
        i = iown[idx]
        j = ioth[idx]
        if i == j:
            continue

        dbconf.iconf[i].append(idx)
        dbconf.confpairs.append((traf.id[i], traf.id[j]))

        rng        = dbconf.tcpa[i, j] * traf.gs[i] / nm
        lato, lono = geo.qdrpos(traf.lat[i], traf.lon[i], traf.trk[i], rng)
        alto       = traf.alt[i] + dbconf.tcpa[i, j] * traf.vs[i]

        dbconf.latowncpa.append(lato)
        dbconf.lonowncpa.append(lono)
        dbconf.altowncpa.append(alto)

        dx = (traf.lat[i] - traf.lat[j]) * 111319.
        dy = (traf.lon[i] - traf.lon[j]) * 111319.

        hdist2 = dx**2 + dy**2
        hLOS   = hdist2 < dbconf.R**2
        vdist  = abs(traf.alt[i] - traf.alt[j])
        vLOS   = vdist < dbconf.dh
        LOS    = (hLOS & vLOS)

        # Add to Conflict and LOSlist, to count total conflicts and LOS

        # NB: if only one A/C detects a conflict, it is also added to these lists
        combi = str(traf.id[i]) + " " + str(traf.id[j])
        combi2 = str(traf.id[j]) + " " + str(traf.id[i])

        experimenttime = simt > 2100 and simt < 5700  # These parameters may be
        # changed to count only conflicts within a given expirement time window

        if combi not in dbconf.conflist_all and combi2 not in dbconf.conflist_all:
            dbconf.conflist_all.append(combi)

        if combi not in dbconf.conflist_exp and combi2 not in dbconf.conflist_exp and experimenttime:
            dbconf.conflist_exp.append(combi)

        if combi not in dbconf.conflist_now and combi2 not in dbconf.conflist_now:
            dbconf.conflist_now.append(combi)

        if LOS:
            if combi not in dbconf.LOSlist_all and combi2 not in dbconf.LOSlist_all:
                dbconf.LOSlist_all.append(combi)
                dbconf.LOSmaxsev.append(0.)
                dbconf.LOShmaxsev.append(0.)
                dbconf.LOSvmaxsev.append(0.)

            if combi not in dbconf.LOSlist_exp and combi2 not in dbconf.LOSlist_exp and experimenttime:
                dbconf.LOSlist_exp.append(combi)

            if combi not in dbconf.LOSlist_now and combi2 not in dbconf.LOSlist_now:
                dbconf.LOSlist_now.append(combi)

            # Now, we measure intrusion and store it if it is the most severe
            Ih = 1.0 - np.sqrt(hdist2) / dbconf.R
            Iv = 1.0 - vdist / dbconf.dh
            severity = min(Ih, Iv)

            try:  # Only continue if combi is found in LOSlist (and not combi2)
                idx = dbconf.LOSlist_all.index(combi)
            except:
                idx = -1

            if idx >= 0:
                if severity > dbconf.LOSmaxsev[idx]:
                    dbconf.LOSmaxsev[idx]  = severity
                    dbconf.LOShmaxsev[idx] = Ih
                    dbconf.LOSvmaxsev[idx] = Iv

    # Convert to numpy arrays for vectorisation
    dbconf.latowncpa = np.array(dbconf.latowncpa)
    dbconf.lonowncpa = np.array(dbconf.lonowncpa)
    dbconf.altowncpa = np.array(dbconf.altowncpa)

    # Calculate whether ASAS or A/P commands should be followed
    APorASAS(dbconf, traf)


def APorASAS(dbconf, traf):
    """ Decide for each aircraft in the conflict list whether the ASAS
        should be followed or not, based on if the aircraft pairs passed
        their CPA. """
    dbconf.asasactive.fill(False)

    # Look at all conflicts, also the ones that are solved but CPA is yet to come
    for conflict in dbconf.conflist_all:
        ac1, ac2 = conflict.split(" ")
        id1, id2 = traf.id2idx(ac1), traf.id2idx(ac2)
        if id1 >= 0 and id2 >= 0:
            # Check if conflict is past CPA
            d = np.array([traf.lon[id2] - traf.lon[id1], traf.lat[id2] - traf.lat[id1]])

            t1 = np.radians(traf.trk[id1])
            t2 = np.radians(traf.trk[id2])

            # write velocities as vectors and find relative velocity vector
            v1 = np.array([np.sin(t1) * traf.tas[id1], np.cos(t1) * traf.tas[id1]])
            v2 = np.array([np.sin(t2) * traf.tas[id2], np.cos(t2) * traf.tas[id2]])

            if np.dot(d, v2 - v1) < 0:
                # Aircraft haven't passed their CPA: must follow their ASAS
                dbconf.asasactive[id1] = True
                dbconf.asasactive[id2] = True
