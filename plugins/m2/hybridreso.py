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
    reso = hybridreso() 

    # Configuration parameters
    config = {
        # The name of your plugin. Keep it the same as the class
        'plugin_name':     'hybridreso',

        # The type of this plugin.  For now, only simulation plugins are possible.
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
        
        # Loop through each conflict, determine the resolution method, and set 
        # the resolution for the asas module
        for conflict in conf.confpairs:
            
            #idx of ownship and intruder
            idxown, idxint = traf.id2idx(conflict)
            
            # TODO: Check for multi-aircraft conflicts 
            
            # determine priority of the aircraft in conflict
            ownshipResolves = self.priorityChecker(idxown, idxint)
            
        return newtrk, newgs, newvs, newalt
    
    def priorityChecker(self, idxown, idxint):
        'Determines if the ownship has lower priority and therefore has to resolve the conflict'
        
        prioOwn = traf.priority[idxown]
        prioInt = traf.priority[idxint]
        
        # Compare the priority of ownship and intruder
        if prioOwn < prioInt: # if ownship has lower priority, then it resolves
            return True
        elif prioOwn == prioInt: # if both drones have the same priority, the callsign breaks the deadlock
            # get number in the callsign of the ownship and intruder
            numberOwn = int("".join([str(elem) for elem in [int(word) for word in traf.id[idxown] if word.isdigit()]]))
            numberInt = int("".join([str(elem) for elem in [int(word) for word in traf.id[idxint] if word.isdigit()]]))
            
            # The aircraft if the the higher callsign has lower priority, and therefore has to resolve
            if numberOwn > numberInt:
                return True
            else:
                return False
        else:# if the ownship has higher priority, then it does not resolve
            return False
        