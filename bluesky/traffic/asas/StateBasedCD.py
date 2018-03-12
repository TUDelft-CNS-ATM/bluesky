''' State-based conflict detection. '''
import numpy as np
from bluesky.tools import geo
from bluesky.tools.aero import nm


def detect(ownship, intruder, RPZ, HPZ, tlookahead):
    ''' Conflict detection between ownship (traf) and intruder (traf/adsb).'''

    # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
    I = np.eye(ownship.ntraf)

    # Horizontal conflict ------------------------------------------------------

    # qdlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
    qdr, dist = geo.qdrdist_matrix(np.mat(ownship.lat), np.mat(ownship.lon),
                                   np.mat(intruder.lat), np.mat(intruder.lon))

    # Convert back to array to allow element-wise array multiplications later on
    # Convert to meters and add large value to own/own pairs
    qdr = np.array(qdr)
    dist = np.array(dist) * nm + 1e9 * I

    # Calculate horizontal closest point of approach (CPA)
    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)  # is pos j rel to i
    dy = dist * np.cos(qdrrad)  # is pos j rel to i

    # Intruder track angle and speed
    inttrkrad = np.radians(intruder.trk)
    intu = intruder.gs * np.cos(inttrkrad).reshape((1, ownship.ntraf))  # m/s
    intv = intruder.gs * np.sin(inttrkrad).reshape((1, ownship.ntraf))  # m/s

    du = ownship.gsnorth.reshape((1, ownship.ntraf)) - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
    dv = ownship.gseast.reshape((1, ownship.ntraf)) - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
    vrel = np.sqrt(dv2)

    tcpa = -(du * dx + dv * dy) / dv2 + 1e9 * I

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = dist * dist - tcpa * tcpa * dv2

    # Check for horizontal conflict
    R2 = RPZ * RPZ
    swhorconf = dcpa2 < R2  # conflict or not

    # Calculate times of entering and leaving horizontal conflict
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
    dtinhor = dxinhor / vrel

    tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
    touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf

    # Vertical conflict --------------------------------------------------------

    # Vertical crossing of disk (-dh,+dh)
    dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
        intruder.alt.reshape((1, ownship.ntraf)).T

    dvs = ownship.vs.reshape(1, ownship.ntraf) - \
        intruder.vs.reshape(1, ownship.ntraf).T
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero

    # Check for passing through each others zone
    tcrosshi = (dalt + HPZ) / -dvs
    tcrosslo = (dalt - HPZ) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)

    # Combine vertical and horizontal conflict----------------------------------
    tinconf = np.maximum(tinver, tinhor)
    toutconf = np.minimum(toutver, touthor)

    swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) * \
        (tinconf < tlookahead) * (1.0 - I), dtype=np.bool)

    # --------------------------------------------------------------------------
    # Update conflict lists
    # --------------------------------------------------------------------------

    # Select conflicting pairs: each a/c gets their own record
    confpairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swconfl))]
    # bearing, dist, tcpa, tinconf, toutconf per conflict
    if confpairs:
        swlos = (dist < RPZ) * (dalt < HPZ)
        lospairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swlos))]
        qdr = np.squeeze(np.array(qdr[swconfl]))
        dist = np.squeeze(np.array(dist[swconfl]))
        tcpa = np.squeeze(np.array(tcpa[swconfl]))
        tinconf = np.squeeze(np.array(tinconf[swconfl]))
        toutconf = np.squeeze(np.array(toutconf[swconfl]))
        return confpairs, lospairs, qdr, dist, tcpa, tinconf, toutconf
    return [[]] * 7
