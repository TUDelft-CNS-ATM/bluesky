""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
from random import randint
import numpy as np
import itertools
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union, nearest_points
from shapely.affinity import translate
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
from bluesky.traffic.asas import ConflictResolution
from bluesky.tools.geo import kwikdist, kwikqdrdist
from bluesky.tools.aero import nm, ft


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    # reso = hybridreso()

    # Configuration parameters
    config = {
        # The name of your plugi
        'plugin_name':     'hybridreso',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


class hybridreso(ConflictResolution):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        # All classes deriving from Entity can register lists and numpy arrays
        # that hold per-aircraft data. This way, their size is automatically
        # updated when aircraft are created or deleted in the simulation.
            
            
    def resolve(self, conf, ownship, intruder):
        '''We want to only solve in the velocity direction while still following the heading
        given by the autopilot. For each ownship, it calculates the minimum or maximum velocity
        needed to avoid all intruders. It then applies the solution closest to the current velocity.
        If there is no solution, it should then apply speed 0 by default, and the aircraft stops.'''
        
        # note 'conf' is the CD object
        # only update hdgactive/tasactive/vsactive/trkactive to be True only for the aircraft that need to resolve
        # this resolve function should have the following four outputs: newtrk, newgs, newvs, newalt
               
             
        # Make a copy of traffic data, ground speed and alt. 
        # For gs and trk, determine the new values for the resolving aircraft using the appropriate reso method
        newgs  = np.copy(traf.gs)
        newalt = np.copy(traf.ap.alt)
        
        # Speed based, and 2D, for now.
        newtrk = np.copy(traf.ap.trk)
        newvs = np.copy(traf.vs)
        
        # import pdb
        # pdb.set_trace()
        
        # call the intent filter
        self.intentFilter(conf, ownship, intruder)
        
        # # set the trk and vs active flags to flase for all traffic
        # self.trkactive = np.array([False]*traf.ntraf)
        # self.vsactive  = np.array([False]*traf.ntraf)
        
        
        # Iterate over aircraft in conflict and remove the conflcits that are automatically resolved 
        # for idx in list(itertools.compress(range(len(traf.cr.active)), traf.cr.active)):
            
            
            
            # Find the pairs in which IDX is involved in a conflict
            # idx_pairs = self.pairs(conf, ownship, intruder, idx)
            
            
            
            
            
            # Find solution for aircraft 'idx'. 
            # TODO: make hybrid airborne resolution functions that will compute newgs 
            #       and newalt and also set the self.tasactive and self.altactive for the aircraft that is resolving
            # gs_new, vs_new = self.SpeedBased(conf, ownship, intruder, idx, idx_pairs) # example from Anderi
            
            # Write the new velocity of aircraft 'idx' to traffic data
            # newgs[idx] = gs_new    
            # newvs[idx] = vs_new    
        
        
        
        
        
        return newtrk, newgs, newvs, newalt
    
    def intentFilter(self, conf, ownship, intruder):
        '''Function to check and remove conflicts from the conflict lists
           if such a conflict is automatically so is automatically solved by the routes of the conf '''
           
        # dict to store the the idxs of the aircraft to change their active status
        
        
        changeactive = dict()
        
        for conflict in conf.confpairs:
            idxown, idxint = traf.id2idx(conflict)
            
            own_intent, own_target_alt = ownship.intent[idxown] # TODO: figure out how to import intent
            intruder_intent, intruder_target_alt = intruder.intent[idxint] # TODO
            # Find closest points between the two intent paths
            pown, pint = nearest_points(own_intent, intruder_intent)
            # Find the distance between the points
            point_distance = kwikdist(pown.y, pown.x, pint.y, pint.x) * nm #[m]
            # Also do vertical intent
            # Difference between own altitude and intruder target
            diff = ownship.alt[idxint] - intruder_target_alt
            
            # minimum horizontal separation 
            rpz = (conf.rpz[idxown]+conf.rpz[idxint])
            
            
            # Basically, there are three conditions to be met in order to skip
            # a conflict due to intent:
            # 1. The minimum distance between the horizontal intent lines is greater than r;
            # 2. The difference between the current altitude and the target altitude of the 
            # intruder is greater than the vertical separation margin;
            # 3. The altitude difference and vertical velocity of the intruder have the same sign.
            # This means that if the aircraft is coming from above (negative), and the altitude difference
            # is positive (thus target altitude is below ownship), then their paths will intersect. 
            if (point_distance < rpz ) and (conf.hpz[idxown] >= abs(diff)):
                # if this is a real conflict, set it to active
                changeactive[idxown] = True
                changeactive[idxint] = True

            else:
                # if the intent resolves the conflict, then remove this conflict from the conflict list and set active to False
                conf.confpairs = conf.confpairs.remove(conflict)
                if conflict in conf.confpairs_unique:
                    conf.confpairs = conf.confpairs_unique.remove(conflict)
                if conflict in conf.confpairs_all:
                    conf.confpairs = conf.confpairs_all.remove(conflict)
                # change the active status to False if resolved by intent
                changeactive[idxown] = changeactive.get(idxown, False)
                changeactive[idxint] = changeactive.get(idxint, False)

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            traf.cr.active[idx] = active
            conf.inconf[idx] = active
        
