""" The hybridcd plugin overrides the default statebased CD. It is a combination
of statebased CD with an intent filter at the end. The aim is to reduce the number of 
flase positives using intent. """
import numpy as np
import copy
from shapely.ops import nearest_points

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf  #, core, stack, settings, navdb, sim, scr, tools
from bluesky.traffic.asas import ConflictDetection
from bluesky.tools.geo import kwikdist
from bluesky.tools import geo
from bluesky.tools.aero import nm #, ft

def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    cd = hybridcd()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'hybridcd',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class hybridcd(ConflictDetection):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        
    def detect(self, ownship, intruder, rpz, hpz, dtlookahead):
        
        ###### First Do Statebased CD. This is copy-pasted from statebased.py ######
        
        # Identity matrix of order ntraf: avoid ownship-ownship detected conflicts
        I = np.eye(ownship.ntraf)

        # Horizontal conflict ------------------------------------------------------

        # qdrlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
        qdr, dist = geo.kwikqdrdist_matrix(np.asmatrix(ownship.lat), np.asmatrix(ownship.lon),
                                    np.asmatrix(intruder.lat), np.asmatrix(intruder.lon))

        # Convert back to array to allow element-wise array multiplications later on
        # Convert to meters and add large value to own/own pairs
        qdr = np.asarray(qdr)
        dist = np.asarray(dist) * nm + 1e9 * I

        # Calculate horizontal closest point of approach (CPA)
        qdrrad = np.radians(qdr)
        dx = dist * np.sin(qdrrad)  # is pos j rel to i
        dy = dist * np.cos(qdrrad)  # is pos j rel to i

        # Ownship track angle and speed
        owntrkrad = np.radians(ownship.trk)
        ownu = ownship.gs * np.sin(owntrkrad).reshape((1, ownship.ntraf))  # m/s
        ownv = ownship.gs * np.cos(owntrkrad).reshape((1, ownship.ntraf))  # m/s

        # Intruder track angle and speed
        inttrkrad = np.radians(intruder.trk)
        intu = intruder.gs * np.sin(inttrkrad).reshape((1, ownship.ntraf))  # m/s
        intv = intruder.gs * np.cos(inttrkrad).reshape((1, ownship.ntraf))  # m/s

        du = ownu - intu.T  # Speed du[i,j] is perceived eastern speed of i to j
        dv = ownv - intv.T  # Speed dv[i,j] is perceived northern speed of i to j

        dv2 = du * du + dv * dv
        dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
        vrel = np.sqrt(dv2)

        tcpa = -(du * dx + dv * dy) / dv2 + 1e9 * I

        # Calculate distance^2 at CPA (minimum distance^2)
        dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

        # Check for horizontal conflict
        # RPZ can differ per aircraft, get the largest value per aircraft pair
        rpz = np.asarray(np.maximum(np.asmatrix(rpz), np.asmatrix(rpz).transpose()))
        R2 = rpz * rpz
        swhorconf = dcpa2 < R2  # conflict or not

        # Calculate times of entering and leaving horizontal conflict
        dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
        dtinhor = dxinhor / vrel

        tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
        touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf

        # Vertical conflict --------------------------------------------------------

        # Vertical crossing of disk (-dh,+dh)
        dalt = ownship.alt.reshape((1, ownship.ntraf)) - \
            intruder.alt.reshape((1, ownship.ntraf)).T  + 1e9 * I

        dvs = ownship.vs.reshape(1, ownship.ntraf) - \
            intruder.vs.reshape(1, ownship.ntraf).T
        dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero

        # Check for passing through each others zone
        # hPZ can differ per aircraft, get the largest value per aircraft pair
        hpz = np.asarray(np.maximum(np.asmatrix(hpz), np.asmatrix(hpz).transpose()))
        tcrosshi = (dalt + hpz) / -dvs
        tcrosslo = (dalt - hpz) / -dvs
        tinver = np.minimum(tcrosshi, tcrosslo)
        toutver = np.maximum(tcrosshi, tcrosslo)
        
        # NEW: determine if each aircraft has a vertical conflcit
        swverconf = np.any((tinver<=toutver)*(toutver>0.0)*(tinver<np.asmatrix(dtlookahead).T)*(1.0-I),1)
        

        # Combine vertical and horizontal conflict----------------------------------
        tinconf = np.maximum(tinver, tinhor)
        toutconf = np.minimum(toutver, touthor)

        swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) *
                           np.asarray(tinconf < np.asmatrix(dtlookahead).T) * (1.0 - I), dtype=np.bool)

        # --------------------------------------------------------------------------
        # Update conflict lists
        # --------------------------------------------------------------------------
        # Ownship conflict flag and max tCPA
        inconf = np.any(swconfl, 1)
        tcpamax = np.max(tcpa * swconfl, 1)

        # Select conflicting pairs: each a/c gets their own record
        confpairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swconfl))]
        swlos = (dist < rpz) * (np.abs(dalt) < hpz)
        lospairs = [(ownship.id[i], ownship.id[j]) for i, j in zip(*np.where(swlos))]
        
        
        ####################### Second do intent filter #######################
        if traf.swintent:
            confpairs, inconf = self.intentFilter(traf.cd, confpairs, inconf, ownship, intruder, swverconf)
        
        ########## Finaly return with the filtered confpairs and inconf! ##########

        return confpairs, lospairs, inconf, tcpamax, \
            qdr[swconfl], dist[swconfl], np.sqrt(dcpa2[swconfl]), \
                tcpa[swconfl], tinconf[swconfl]
                
    
    def intentFilter(self, conf, confpairs, inconf, ownship, intruder, swverconf):
        '''Function to check and remove conflicts from the confpairs and inconf lists
           if such a conflict is automatically solved by the intended routes of the aircraft '''
           
        # dict to store the the idxs of the aircraft to change their active status
        changeactive = dict()
        
        # make a deep copy of confpairs in order to loop through and delete from conf.confpairs
        conflicts = copy.deepcopy(confpairs) 
        
        # loop through each conflict and remove conflict if intent resolves conflict
        for conflict in conflicts:
            
            #idx of ownship and intruder
            idxown, idxint = traf.id2idx(conflict)
            
            # minimum horizontal separation 
            rpz = max(conf.rpz[idxown],conf.rpz[idxint])#*1.05
            hpz = max(conf.hpz[idxown],conf.hpz[idxint])
            
            # get the intents of ownship and intruder. This is calculated in the intent plugin.
            own_intent, own_target_alt = ownship.intent[idxown] 
            intruder_intent, intruder_target_alt = intruder.intent[idxint] 
            
            # Find the nearest point in the two line strings
            pown, pint = nearest_points(own_intent, intruder_intent)
            
            # Find the distance between the points
            point_distance = kwikdist(pown.y, pown.x, pint.y, pint.x) * nm #[m]
            
            # Also do vertical intent
            # Difference between own altitude and intruder target
            fpown = traf.flightphase[idxown]
            fpint = traf.flightphase[idxint]
            
            if fpown != fpint:
                diff = own_target_alt - intruder_target_alt
                verticalCondition = hpz >= abs(diff)
            else:
                verticalCondition = swverconf[idxown]      
                
            # Basically, there are two conditions to be met in order to skip
            # a conflict due to intent:
            # 1. The minimum distance between the horizontal intent lines is greater than r;
            # 2. The difference between the current altitude and the target altitude of the 
            # intruder is greater than the vertical separation margin;
            if (point_distance < rpz ) and verticalCondition:
                # if this is a real conflict, set it to active to True
                changeactive[idxown] = True
                changeactive[idxint] = True
            else:
                # if the intent resolves the conflict, then remove this conflict 
                # from the conflict lists and set active to False
                confpairs.remove(conflict)
                # if set(conflict) in conf.confpairs_unique:
                #     conf.confpairs_unique.remove(set(conflict))
                # if set(conflict) in conf.confpairs_all:
                #     conf.confpairs_all.remove(set(conflict))
                changeactive[idxown] = changeactive.get(idxown, False)
                changeactive[idxint] = changeactive.get(idxint, False)
                
        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            # traf.cr.active[idx] = active
            inconf[idx] = active
        
        return confpairs, inconf
