"""
State-based conflict detection


"""
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm


def detect(asas, traf, simt):
    if not asas.swasas:
        return

    # Reset lists before new CD
    asas.iconf        = [[] for ac in range(traf.ntraf)]
    asas.nconf        = 0
    asas.confpairs    = []
    asas.latowncpa    = []
    asas.lonowncpa    = []
    asas.altowncpa    = []

    asas.LOSlist_now  = []
    asas.conflist_now = []

    # Horizontal conflict ---------------------------------------------------------

    # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
    qdlst = geo.qdrdist_matrix(np.mat(traf.lat), np.mat(traf.lon),
                               np.mat(traf.adsb.lat), np.mat(traf.adsb.lon))

    # Convert results from mat-> array
    asas.qdr  = np.array(qdlst[0])  # degrees
    I           = np.eye(traf.ntraf)  # Identity matric of order ntraf
    asas.dist = np.array(qdlst[1]) * nm + 1e9 * I  # meters i to j

    # Transmission noise
    if traf.adsb.transnoise:
        # error in the determined bearing between two a/c
        bearingerror = np.random.normal(0, traf.adsb.transerror[0], asas.qdr.shape)  # degrees
        asas.qdr += bearingerror
        # error in the perceived distance between two a/c
        disterror = np.random.normal(0, traf.adsb.transerror[1], asas.dist.shape)  # meters
        asas.dist += disterror

    # Calculate horizontal closest point of approach (CPA)
    qdrrad    = np.radians(asas.qdr)
    asas.dx = asas.dist * np.sin(qdrrad)  # is pos j rel to i
    asas.dy = asas.dist * np.cos(qdrrad)  # is pos j rel to i

    trkrad   = np.radians(traf.trk)
    asas.u = traf.gs * np.sin(trkrad).reshape((1, len(trkrad)))  # m/s
    asas.v = traf.gs * np.cos(trkrad).reshape((1, len(trkrad)))  # m/s

    # parameters received through ADSB
    adsbtrkrad = np.radians(traf.adsb.trk)
    adsbu = traf.adsb.gs * np.sin(adsbtrkrad).reshape((1, len(adsbtrkrad)))  # m/s
    adsbv = traf.adsb.gs * np.cos(adsbtrkrad).reshape((1, len(adsbtrkrad)))  # m/s

    du = asas.u - adsbu.T  # Speed du[i,j] is perceived eastern speed of i to j
    dv = asas.v - adsbv.T  # Speed dv[i,j] is perceived northern speed of i to j

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value

    vrel = np.sqrt(dv2)

    asas.tcpa = -(du * asas.dx + dv * asas.dy) / dv2 + 1e9 * I

    # Calculate CPA positions
    # xcpa = asas.tcpa * du
    # ycpa = asas.tcpa * dv

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = asas.dist * asas.dist - asas.tcpa * asas.tcpa * dv2

    # Check for horizontal conflict
    R2 = asas.R * asas.R
    swhorconf = dcpa2 < R2  # conflict or not

    # Calculate times of entering and leaving horizontal conflict
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
    dtinhor = dxinhor / vrel

    tinhor = np.where(swhorconf, asas.tcpa - dtinhor, 1e8)  # Set very large if no conf

    touthor = np.where(swhorconf, asas.tcpa + dtinhor, -1e8)  # set very large if no conf
    # swhorconf = swhorconf*(touthor>0)*(tinhor<asas.dtlook)

    # Vertical conflict -----------------------------------------------------------

    # Vertical crossing of disk (-dh,+dh)
    alt = traf.alt.reshape((1, traf.ntraf))
    adsbalt = traf.adsb.alt.reshape((1, traf.ntraf))
    if traf.adsb.transnoise:
        # error in the determined altitude of other a/c
        alterror = np.random.normal(0, traf.adsb.transerror[2], traf.alt.shape)  # degrees
        adsbalt += alterror

    asas.dalt = alt - adsbalt.T


    vs = traf.vs.reshape(1, len(traf.vs))


    avs = traf.adsb.vs.reshape(1, len(traf.adsb.vs))

    dvs = vs - avs.T

    # Check for passing through each others zone
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero
    tcrosshi = (asas.dalt + asas.dh) / -dvs
    tcrosslo = (asas.dalt - asas.dh) / -dvs

    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)

    # Combine vertical and horizontal conflict-------------------------------------
    asas.tinconf = np.maximum(tinver, tinhor)

    asas.toutconf = np.minimum(toutver, touthor)

    swconfl = swhorconf * (asas.tinconf <= asas.toutconf) * \
        (asas.toutconf > 0.) * (asas.tinconf < asas.dtlookahead) \
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
    asas.nconf        = len(confidxs[0])

    for idx in range(asas.nconf):
        i = iown[idx]
        j = ioth[idx]
        if i == j:
            continue

        asas.iconf[i].append(idx)
        asas.confpairs.append((traf.id[i], traf.id[j]))

        rng        = asas.tcpa[i, j] * traf.gs[i] / nm
        lato, lono = geo.qdrpos(traf.lat[i], traf.lon[i], traf.trk[i], rng)
        alto       = traf.alt[i] + asas.tcpa[i, j] * traf.vs[i]

        asas.latowncpa.append(lato)
        asas.lonowncpa.append(lono)
        asas.altowncpa.append(alto)

        dx = (traf.lat[i] - traf.lat[j]) * 111319.
        dy = (traf.lon[i] - traf.lon[j]) * 111319.

        hdist = asas.dist[i,j]
        hLOS   = hdist < asas.R
        vdist  = abs(traf.alt[i] - traf.alt[j])
        vLOS   = vdist < asas.dh
        LOS    = (hLOS & vLOS)

        # Add to Conflict and LOSlist, to count total conflicts and LOS

        # NB: if only one A/C detects a conflict, it is also added to these lists
        combi = str(traf.id[i]) + " " + str(traf.id[j])
        combi2 = str(traf.id[j]) + " " + str(traf.id[i])

        experimenttime = simt > 2100 and simt < 5700  # These parameters may be
        # changed to count only conflicts within a given expirement time window

        if combi not in asas.conflist_all and combi2 not in asas.conflist_all:
            asas.conflist_all.append(combi)

        if combi not in asas.conflist_exp and combi2 not in asas.conflist_exp and experimenttime:
            asas.conflist_exp.append(combi)

        if combi not in asas.conflist_now and combi2 not in asas.conflist_now:
            asas.conflist_now.append(combi)

        if LOS:
            if combi not in asas.LOSlist_all and combi2 not in asas.LOSlist_all:
                asas.LOSlist_all.append(combi)
                asas.LOSmaxsev.append(0.)
                asas.LOShmaxsev.append(0.)
                asas.LOSvmaxsev.append(0.)

            if combi not in asas.LOSlist_exp and combi2 not in asas.LOSlist_exp and experimenttime:
                asas.LOSlist_exp.append(combi)

            if combi not in asas.LOSlist_now and combi2 not in asas.LOSlist_now:
                asas.LOSlist_now.append(combi)

            # Now, we measure intrusion and store it if it is the most severe
            Ih = 1.0 - hdist / asas.R
            Iv = 1.0 - vdist / asas.dh
            severity = min(Ih, Iv)

            try:  # Only continue if combi is found in LOSlist (and not combi2)
                idx = asas.LOSlist_all.index(combi)
            except:
                idx = -1

            if idx >= 0:
                if severity > asas.LOSmaxsev[idx]:
                    asas.LOSmaxsev[idx]  = severity
                    asas.LOShmaxsev[idx] = Ih
                    asas.LOSvmaxsev[idx] = Iv

    # Convert to numpy arrays for vectorisation
    asas.latowncpa = np.array(asas.latowncpa)
    asas.lonowncpa = np.array(asas.lonowncpa)
    asas.altowncpa = np.array(asas.altowncpa)

    # Calculate whether ASAS or A/P commands should be followed
    ResumeNav(asas, traf)


def ResumeNav(asas, traf):
    """ Decide for each aircraft in the conflict list whether the ASAS
        should be followed or not, based on if the aircraft pairs passed
        their CPA. """
    asas.active.fill(False)

    # Look at all conflicts, also the ones that are solved but CPA is yet to come
    for conflict in list(asas.conflist_all):
        ac1, ac2 = conflict.split(" ")
        id1, id2 = traf.id2idx(ac1), traf.id2idx(ac2)
        if id1 >= 0 and id2 >= 0:
            # Check if conflict is past CPA
            d = np.array([traf.lon[id2] - traf.lon[id1], traf.lat[id2] - traf.lat[id1]])

            # write velocities as vectors
            v1 = np.array([traf.gseast[id1], traf.gsnorth[id1]])
            v2 = np.array([traf.gseast[id2], traf.gsnorth[id2]])

            # Compute pastCPA
            pastCPA = np.dot(d,v2-v1)>0.

            # hLOS:
            # Aircraft should continue to resolve until there is no horizontal
            # LOS. This is particularly relevant when vertical resolutions
            # are used.
            hdist = asas.dist[id1,id2]
            hLOS   = hdist < asas.R

            # Bouncing conflicts:
            # If two aircraft are getting in and out of conflict continously,
            # then they it is a bouncing conflict. ASAS should stay active until
            # the bouncing stops.
            bouncingConflict = (abs(traf.trk[id1] - traf.trk[id2]) < 30.) & (hdist<asas.Rm**2)

            # Decide if conflict is over or not.
            # If not over, turn active to true.
            # If over, then initiate recovery
            if not pastCPA or hLOS or bouncingConflict:
                # Aircraft haven't passed their CPA: must follow their ASAS
                asas.active[id1] = True
                asas.active[id2] = True

            else:
                # Waypoint recovery after conflict
                # Find the next active waypoint and send the aircraft to that
                # waypoint.
                iwpid1 = traf.ap.route[id1].findact(id1)
                if iwpid1 != -1: # To avoid problems if there are no waypoints
                    traf.ap.route[id1].direct(id1, traf.ap.route[id1].wpname[iwpid1])
                iwpid2 = traf.ap.route[id2].findact(id2)
                if iwpid2 != -1: # To avoid problems if there are no waypoints
                    traf.ap.route[id2].direct(id2, traf.ap.route[id2].wpname[iwpid2])

                # If conflict is solved, remove it from conflist_all list
                # This is so that if a conflict between this pair of aircraft
                # occurs again, then that new conflict should be detected, logged
                # and solved (if reso is on)
                asas.conflist_all.remove(conflict)

        # If aircraft id1 cannot be found in traffic because it has finished its
        # flight (and has been deleted), start trajectory recovery for aircraft id2
        # And remove the conflict from the conflict_all list
        elif id1 < 0 and id2 >= 0:
             iwpid2 = traf.ap.route[id2].findact(id2)
             if iwpid2 != -1: # To avoid problems if there are no waypoints
                 traf.ap.route[id2].direct(id2, traf.ap.route[id2].wpname[iwpid2])
             asas.conflist_all.remove(conflict)

        # If aircraft id2 cannot be found in traffic because it has finished its
        # flight (and has been deleted) start trajectory recovery for aircraft id1
        # And remove the conflict from the conflict_all list
        elif id2 < 0 and id1 >= 0:
            iwpid1 = traf.ap.route[id1].findact(id1)
            if iwpid1 != -1: # To avoid problems if there are no waypoints
                traf.ap.route[id1].direct(id1, traf.ap.route[id1].wpname[iwpid1])
            asas.conflist_all.remove(conflict)

        # if both ids are unknown, then delete this conflict, because both aircraft
        # have completed their flights (and have been deleted)
        else:
            asas.conflist_all.remove(conflict)
