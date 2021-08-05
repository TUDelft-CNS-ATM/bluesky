""" The hybridreso plugin performs tactical airborne conflict resolution in the
    Hybrid concept of the Metropolis 2 project.
    Created by: Emmanuel    
    Date: 29 July 2021
"""
import numpy as np
import copy
from shapely.ops import nearest_points
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf  # core, stack, settings, navdb, sim, scr, tools
from bluesky.traffic.asas import ConflictResolution
from bluesky.tools.geo import kwikdist
from bluesky.tools.aero import nm #, ft


def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity. Seems to work even if you don't do this.
    # reso = hybridreso() 

    # Configuration parameters
    config = {
        # The name of your plugin. Keep it the same as the class
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
            
    def resolve(self, conf, ownship, intruder):
        ''' This resolve function will override the default resolution of resolution.py
            It should return the gs, vs, alt and trk that the conflicting aircraft should 
            fly to return the conflict. The hypri'''
        # note 'conf' is the CD object --> traf.cd
        # only update hdgactive/tasactive/vsactive/trkactive to be True only for the aircraft that need to resolve
        # this resolve function should have the following four outputs: newtrk, newgs, newvs, newalt
             
        # Make a copy of the traffic gs, alt, trk and vs. These are the outputs of this function.
        # The airborne algorithms of the hybrid algorithms will only affect alt and gs
        newgs  = np.copy(traf.gs)
        newalt = np.copy(traf.ap.alt)
        newtrk = np.copy(traf.ap.trk)
        newvs  = np.copy(traf.vs)
        
        # Step 1: Intent filter to remove conflicts that will automatically be 
        #         resolved during the look-ahead time by aircraft following their current routes 
        self.intentFilter(conf, ownship, intruder)
        
        # Step 2: Loop through each conflict, determine the correct reso algorithm (flowchart)
        #         and call the correct reso algorithm. The output of the algorithms should be used
        #         modify newgs and newalt for the resolving aircrafts  
                
        
        return newtrk, newgs, newvs, newalt
    
    def intentFilter(self, conf, ownship, intruder):
        '''Function to check and remove conflicts from the conflict lists
           if such a conflict is automatically so is automatically solved by the routes of the conf '''
           
        # dict to store the the idxs of the aircraft to change their active status
        changeactive = dict()
        
        # make a deep copy of confpairs in order to loop through and delete from conf.confpairs
        confpairs = copy.deepcopy(conf.confpairs) 
        
        # loop through each conflict and remove conflicts if intent resolves conflict
        for conflict in confpairs:
            
            #idx of ownship and intruder
            idxown, idxint = traf.id2idx(conflict)
            
            # get the intents of ownship and intruder. This is calculated in the intent plugin.
            own_intent, own_target_alt = ownship.intent[idxown] 
            intruder_intent, intruder_target_alt = intruder.intent[idxint] 
            
            # Find the nearest point in the two line strings
            pown, pint = nearest_points(own_intent, intruder_intent)
            
            # Find the distance between the points
            point_distance = kwikdist(pown.y, pown.x, pint.y, pint.x) * nm #[m]
            
            # Also do vertical intent
            # Difference between own altitude and intruder target
            diff = own_target_alt - intruder_target_alt
            
            # minimum horizontal separation 
            rpz = (conf.rpz[idxown]+conf.rpz[idxint])
            
            # Basically, there are three conditions to be met in order to skip
            # a conflict due to intent:
            # 1. The minimum distance between the horizontal intent lines is greater than r;
            # 2. The difference between the current altitude and the target altitude of the 
            # intruder is greater than the vertical separation margin;
            if (point_distance < rpz ) and (conf.hpz[idxown] >= abs(diff)):
                # if this is a real conflict, set it to active to True
                changeactive[idxown] = True
                changeactive[idxint] = True

            else:
                # if the intent resolves the conflict, then remove this conflict 
                # from the conflict lists and set active to False
                conf.confpairs.remove(conflict)
                if set(conflict) in conf.confpairs_unique:
                    conf.confpairs_unique.remove(set(conflict))
                if set(conflict) in conf.confpairs_all:
                    conf.confpairs_all.remove(set(conflict))
                changeactive[idxown] = changeactive.get(idxown, False)
                changeactive[idxint] = changeactive.get(idxint, False)
                
        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            traf.cr.active[idx] = active
            conf.inconf[idx] = active
            