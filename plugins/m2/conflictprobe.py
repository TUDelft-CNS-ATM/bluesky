import numpy as np
from shapely.ops import nearest_points

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf  # core, stack, settings, navdb, sim, scr, tools
from bluesky.tools.aero import nm#, ft
from bluesky.tools import geo


def conflictProbe(ownship, intruder, idxown, idxint=-9999, dtlook=traf.cd.dtlookahead, intent=False, targetLat=9999, targetLon=9999, targetAlt=-9999, intAlt=-9999, targetVs=-9999, targetGs=9999, targetTrk=9999):
    'Returns True if a conflict would occur if the ownship adopts the input target states'
    'Here the optional argument idxint is used to artifically supress a conflict with this intruder'
    
    # Get the separation requirements
    rpz = traf.cd.rpz
    hpz = traf.cd.hpz
    
    # get the correct ownship parameters for idxown and replace with target states as required
    ownlat    = ownship.lat[idxown] if targetLat == 9999 else targetLat
    ownlon    = ownship.lon[idxown] if targetLon == 9999 else targetLon
    owntrk    = ownship.trk[idxown] if targetTrk == 9999 else targetTrk
    owngs     = ownship.gs[idxown]  if targetGs  == 9999 else targetGs
    ownalt    = ownship.alt[idxown] if targetAlt == -9999 else targetAlt
    intentalt = ownship.alt[idxown] if intAlt == -9999 else intAlt
    ownvs     = ownship.vs[idxown]  if targetVs  == -9999 else targetVs
    
    
    ###### First Do Statebased CD. This is mostly copy-pasted from statebased.py ######
    
    # Horizontal conflict ------------------------------------------------------

    # qdrlst is for [i,j] qdr from i to j, from perception of ADSB and own coordinates
    qdr, dist = geo.kwikqdrdist(ownlat, ownlon, intruder.lat, intruder.lon)
    
    # Convert to meters
    dist = dist * nm 

    # Calculate horizontal closest point of approach (CPA)
    qdrrad = np.radians(qdr)
    dx = dist * np.sin(qdrrad)  # is pos j rel to i
    dy = dist * np.cos(qdrrad)  # is pos j rel to i

    # Ownship track angle and speed
    owntrkrad = np.radians(owntrk)
    ownu = owngs * np.sin(owntrkrad)  # m/s
    ownv = owngs * np.cos(owntrkrad)  # m/s

    # Intruder track angle and speed
    inttrkrad = np.radians(intruder.trk)
    intu = intruder.gs * np.sin(inttrkrad)  # m/s
    intv = intruder.gs * np.cos(inttrkrad)  # m/s

    du = ownu - intu  # Speed du[i,j] is perceived eastern speed of i to j
    dv = ownv - intv  # Speed dv[i,j] is perceived northern speed of i to j

    dv2 = du * du + dv * dv
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
    vrel = np.sqrt(dv2)

    tcpa = -(du * dx + dv * dy) / dv2 

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)

    # Check for horizontal conflict
    # RPZ can differ per aircraft, get the largest value per aircraft pair
    rpz = np.maximum(rpz[idxown], rpz)
    R2 = rpz * rpz
    swhorconf = dcpa2 < R2  # conflict or not

    # Calculate times of entering and leaving horizontal conflict
    dxinhor = np.sqrt(np.maximum(0., R2 - dcpa2))  # half the distance travelled inzide zone
    dtinhor = dxinhor / vrel

    tinhor = np.where(swhorconf, tcpa - dtinhor, 1e8)  # Set very large if no conf
    touthor = np.where(swhorconf, tcpa + dtinhor, -1e8)  # set very large if no conf
    
    
    # Vertical conflict --------------------------------------------------------
    
    # Vertical crossing of disk (-dh,+dh)
    dalt = ownalt - intruder.alt 

    dvs = ownvs - intruder.vs
    dvs = np.where(np.abs(dvs) < 1e-6, 1e-6, dvs)  # prevent division by zero

    # Check for passing through each others zone
    # hPZ can differ per aircraft, get the largest value per aircraft pair
    hpz = np.maximum(hpz[idxown], hpz)
    tcrosshi = (dalt + hpz) / -dvs
    tcrosslo = (dalt - hpz) / -dvs
    tinver = np.minimum(tcrosshi, tcrosslo)
    toutver = np.maximum(tcrosshi, tcrosslo)
    
    # Combine vertical and horizontal conflict----------------------------------
    tinconf = np.maximum(tinver, tinhor)
    toutconf = np.minimum(toutver, touthor)
    
    swconfl = np.array(swhorconf * (tinconf <= toutconf) * (toutconf > 0.0) * (tinconf < dtlook), dtype=np.bool)
    
    # Ownship can't conflict with itself 
    swconfl[idxown] = False
    
    # If idxint > 0, and there is a conflict between ownship and idxint 
    # then it is assumed that the resolution will resolve the original conflict
    # this is useful when the conflict probe is called to determine the right CR strategy
    if idxint >= 0 :
         swconfl[idxint] = False
         
    ########################## Now do intent ##################################
         
    if intent and traf.swintent: # comment out the traf.swintent to test this using intenttest.scn
        swconfl = intentFilterCP(swconfl, ownship, intruder, idxown, intentalt)
        
    
    ############################# NOW RETURN ################################## 
         
    # determine if ownship will get into conflict with target states with any intruder
    probe = any(swconfl)
    
    return probe

def intentFilterCP(swconfl, ownship, intruder, idxown, intentalt):
    '''Function to check and remove conflicts from the swconfl
       if such a conflict is automatically solved by the intended routes of the aircraft '''
       
    # NOTE: THE OWNSHIP INTENT IS ALWAYS CALCULATED USING settings.asas_dtlookahead
    #       REGARDLESS OF THE DTLOOK FOR conflictProbe!!! 
    #       But this should be ok since this function only filters the conflicts 
    #       by state-based with the lower dtlook
    
    # get the horizontal intent of ownship. This is calculated in the intent plugin.
    own_intent, foobar = ownship.intent[idxown]
     
    # set the vertical intent of ownship
    own_target_alt = intentalt
    
    # loop through all the aircraft
    for i in range(len(swconfl)):
        
        # only do intent filter for the intruders that triggered a statebased conflict
        if not swconfl[i]:
            continue
        
        # get the intruder intent
        intruder_intent, intruder_target_alt = intruder.intent[i] 
        
        # Find the nearest point in the two line strings
        pown, pint = nearest_points(own_intent, intruder_intent)
        
        # Find the distance between the points
        point_distance = geo.kwikdist(pown.y, pown.x, pint.y, pint.x) * nm #[m]
        
        # Also do vertical intent
        # Difference between own altitude and intruder target
        diff = own_target_alt - intruder_target_alt
        
        # minimum horizontal separation 
        rpz = (traf.cd.rpz[idxown]+traf.cd.rpz[i])#*1.05
        
        # Basically, there are two conditions to be met in order to skip
        # a conflict due to intent:
        # 1. The minimum distance between the horizontal intent lines is greater than r;
        # 2. The difference between the current altitude and the target altitude of the 
        # intruder is greater than the vertical separation margin;
        if (point_distance < rpz ) and (traf.cd.hpz[idxown] >= abs(diff)):
            # if this is a real conflict, set it to active to True
            swconfl[i] = True
        else:
            # if the intent resolves the conflict, then remove this conflict 
            # from the conflict lists and set active to False
            swconfl[i] = False
            
    return swconfl      
