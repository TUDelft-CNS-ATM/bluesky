""" The hybridreso plugin performs tactical airborne conflict resolution in the
    Hybrid concept of the Metropolis 2 project.
    Created by: Emmanuel    
    Date: 29 July 2021
"""
import numpy as np
import copy

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf, stack  # core, settings, navdb, sim, scr, tools
from bluesky.traffic.asas import ConflictResolution
from bluesky.tools.aero import nm, ft
from plugins.m2.conflictprobe import conflictProbe


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
        newgs  = copy.deepcopy(traf.gs)
        newalt = copy.deepcopy(traf.ap.alt)
        newtrk = copy.deepcopy(traf.ap.trk)
        newvs  = copy.deepcopy(traf.vs)
        
        # Loop through each conflict, determine the resolution method, and set 
        # the resolution for the asas module
        for conflict in conf.confpairs:
            
            #idx of ownship and intruder
            idxown, idxint = traf.id2idx(conflict)
            
            # TODO: Check for multi-aircraft conflicts 
            
            # determine priority of the aircraft in conflict
            ownshipResolves = self.priorityChecker(idxown, idxint)
            
            # if ownship is resolving determine which reso method to use and use it!
            if ownshipResolves:
                
                # determine the ownship and intruder flight phase
                fpown = traf.flightphase[idxown]
                fpint = traf.flightphase[idxint]
                
                # determine if the ownship is in a resolution layer (True if ownship is in a resolution layer)
                rlayerown = traf.aclayername[idxown].lower().count("reso")>0
                
                # determine if the ownship is below the intruder
                belowown = traf.alt[idxown] < traf.alt[idxint] 
                
                # determine if the ownship is above the intruder 
                aboveown = traf.alt[idxown] > traf.alt[idxint]
                
                # Get the max and min vertical speed of ownship (needed for conflict probe)
                vsMinOwn = traf.perf.vsmin[idxown]
                vsMaxOwn = traf.perf.vsmax[idxown]
                
                # determine the look-ahead for the conflict probe
                dtlookup   = np.abs(traf.layerHeight/vsMaxOwn)
                dtlookdown = np.abs(traf.layerHeight/vsMinOwn)
                
                # test the conflict probe
                # probe = conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookup, targetVs=0.0)
                # probe = conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookup, targetVs=vsMaxOwn)#, intent=True, targetAlt=155*ft)
                # print(conflict)
                # print(probe)
                
                ################## START CR Strategy Switch ##################
                
                # ownship and intruder are cruising
                if fpown == 0 and fpint == 0: 
                    if rlayerown:
                        newgs[idxown], traf.cr.tasactive[idxown] = self.reso2(idxown) # use the "speed resolution" strategy
                        # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso2: speed strategy")
                    else:
                        if not conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookup, targetVs=vsMaxOwn, intent=True):
                            newalt[idxown], traf.cr.altactive[idxown] = self.reso1(idxown) # use the "climb into resolution layer" strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso1: climb into resolution layer strategy")
                        else:
                            newgs[idxown], traf.cr.tasactive[idxown] = self.reso2(idxown) # use the speed resolution strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso2: speed strategy")
                            
                            
                # ownship is cruising and intruder is climbing            
                elif fpown == 0 and fpint == 1: 
                    if rlayerown:
                        newgs[idxown], traf.cr.tasactive[idxown] = self.reso2(idxown) # use the "speed resolution" strategy
                        # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso2: speed strategy")
                    else:
                        if not conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookup, targetVs=vsMaxOwn, intent=True):
                            newalt[idxown], newgs[idxown], traf.cr.altactive[idxown], traf.cr.tasactive[idxown] = self.reso5(idxown) # use the climb into resolution layer + speed resolution strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso5: climb into resolution layer + speed resolution strategy")
                        else:
                            newgs[idxown], traf.cr.tasactive[idxown] = self.reso2(idxown) # use the speed resolution strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso2: speed strategy")
                            
                
                # ownship is cruising and intruder is descending
                elif fpown == 0 and fpint == 2: 
                    if rlayerown:
                        newgs[idxown], traf.cr.tasactive[idxown] = self.reso2(idxown) # use the speed resolution strategy
                        # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso2: speed strategy")
                    else:
                        if not conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookup, targetVs=vsMaxOwn, intent=True):
                            newalt[idxown], newgs[idxown], traf.cr.altactive[idxown], traf.cr.tasactive[idxown] = self.reso5(idxown) # use the climb into resolution layer + speed resolution strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso5: climb into resolution layer + speed resolution strategy")
                        else:
                            newgs[idxown], traf.cr.tasactive[idxown] = self.reso2(idxown) # use the speed resolution strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso2: speed strategy")
                
                # ownship is climbing and intruder is cruising
                elif fpown == 1 and fpint == 0: 
                    if rlayerown:
                        newgs[idxown], traf.cr.tasactive[idxown] = self.reso3(idxown) # Hover in the resolution layer strategy
                        # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso5: hover in the resolution layer strategy")
                    else:
                        if not conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookup, targetVs=vsMaxOwn, intent=True):
                            newalt[idxown], newgs[idxown], traf.cr.altactive[idxown], traf.cr.tasactive[idxown] = self.reso6(idxown) # use the climb into resolution layer + hover resolution strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso6: climb into resolution layer + hover resolution strategy")
                        else:
                            newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")
                
                # ownship is climbing and intruder is climbing
                elif fpown == 1 and fpint == 1: 
                    if belowown:
                        if rlayerown:
                            newgs[idxown], traf.cr.tasactive[idxown] = self.reso3(idxown) # Hover in the resolution layer strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso5: hover in the resolution layer strategy")
                        else:
                            if not conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookup, targetVs=vsMaxOwn, intent=True):
                                newalt[idxown], newgs[idxown], traf.cr.altactive[idxown], traf.cr.tasactive[idxown] = self.reso6(idxown) # use the climb into resolution layer + hover resolution strategy
                                # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso6: climb into resolution layer + hover resolution strategy")
                            else:
                                newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                                # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")
                    else:
                        newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                        # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")

                # ownship is climbing and intruder is descending
                elif fpown == 1 and fpint == 2: 
                    newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                    # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")
                    
                # ownship is descending and intruder is cruising
                elif fpown == 2 and fpint == 0: 
                    if rlayerown:
                        newgs[idxown], traf.cr.tasactive[idxown] = self.reso3(idxown) # Hover in the resolution layer strategy
                        # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso5: hover in the resolution layer strategy")
                    else:
                        if not conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookdown, targetVs=vsMinOwn, intent=True):
                            newalt[idxown], newgs[idxown], traf.cr.altactive[idxown], traf.cr.tasactive[idxown] = self.reso7(idxown) # use the descend into resolution layer + hover resolution strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso7: descend into resolution layer + hover resolution strategy")
                        else:
                            newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")
                
                # ownship is descending and intruder is climbing
                elif fpown == 2 and fpint == 1: 
                    newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                    # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")
                
                # ownship is descending and intruder is descending
                elif fpown == 2 and fpint == 2: 
                    if aboveown:
                        if rlayerown:
                            newgs[idxown], traf.cr.tasactive[idxown] = self.reso3(idxown) # Hover in the resolution layer strategy
                            # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso5: hover in the resolution layer strategy")
                        else:
                            if not conflictProbe(ownship, intruder, idxown, idxint, dtlook=dtlookdown, targetVs=vsMinOwn, intent=True):
                                newalt[idxown], newgs[idxown], traf.cr.altactive[idxown], traf.cr.tasactive[idxown] = self.reso7(idxown) # use the descend into resolution layer + hover resolution strategy
                                # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso7: descend into resolution layer + hover resolution strategy")
                            else:
                                newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                                # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")
                    else:
                        newalt[idxown], traf.cr.altactive[idxown] = self.reso4(idxown) # temporarily level off strategy
                        # stack.stack(f"ECHO {traf.id[idxown]} is resolving conflict with {traf.id[idxint]} using reso4: temporarily level off strategy")
                else:
                    print("ERROR: THE FLIGHT PHASE HAS BEEN COMPUTED WORNGLY. CHECK THE flightphase PLUGIN ")
                
                ################### END CR Strategy Switch ###################
            
        return newtrk, newgs, newvs, newalt
    
    
    def priorityChecker(self, idxown, idxint):
        'Determines if the ownship has lower priority and therefore has to resolve the conflict'
        
        # get the priority of the ownship and intruder from traf
        prioOwn = traf.priority[idxown]
        prioInt = traf.priority[idxint]
        
        # Compare the priority of ownship and intruder
        if prioOwn < prioInt: # if ownship has lower priority, then it resolves
            return True
        
        elif prioOwn == prioInt: # if both drones have the same priority, the callsign breaks the deadlock
        
            # get number in the callsign of the ownship and intruder
            numberOwn = int("".join([str(elem) for elem in [int(word) for word in traf.id[idxown] if word.isdigit()]])) # int(traf.id[idxown][1:]) # This is a simpler and faster solution if callsigns are of the format 'D12345'
            numberInt = int("".join([str(elem) for elem in [int(word) for word in traf.id[idxint] if word.isdigit()]])) # int(traf.id[idxint][1:])
            
            # The aircraft if the the higher callsign has lower priority, and therefore has to resolve
            if numberOwn > numberInt:
                return True
            else:
                return False
            
        else:# if the ownship has higher priority, then it does not resolve
            return False
        
    
    def reso1(self, idxown):
        'The climb into resolution layer strategy'
        
        # TODO: Make the drone climb into the correct resolution layer
        newalt = traf.alt[idxown]
        altactive = True
        
        return newalt, altactive
    
    
    def reso2(self, idxown): 
        'The speed resolution strategy'
        
        # TODO: Compute the correct speed to resolve the conflict
        newgs = traf.gs[idxown]
        tasactive = True
        
        return newgs, tasactive
    
    
    def reso3(self, idxown): 
        'The hover in the resolution layer strategy'
        
        # TODO: Make the ownship hover
        newgs = traf.gs[idxown] # 0.0
        tasactive = True
        
        return newgs, tasactive
    
    def reso4(self, idxown): 
        'The temporarily level-off strategy'
        
        # TODO: determine the correct altitude for the drone to level off
        newalt = traf.alt[idxown]
        altactive = True
        
        return newalt, altactive
    
    
    def reso5(self, idxown): 
        'The Climb into resolution layer + speed strategy'
        
        # TODO: determine the correct altitude for the drone to climb to  (maybe call reso1)
        newalt = traf.alt[idxown]
        altactive = True
        
        # TODO: determine the correct the speed the drone should fly (maybe call reso2)
        newgs = traf.gs[idxown]
        tasactive = True
                
        return newalt, newgs, altactive, tasactive
    
        
    def reso6(self, idxown): 
        'The Climb into resolution layer + hover strategy'
        
        # TODO: determine the correct altitude for the drone to climb to  (maybe call reso1)
        newalt = traf.alt[idxown]
        altactive = True
        
        # TODO: make it hover
        newgs = traf.gs[idxown] # 0.0
        tasactive = True
                
        return newalt, newgs, altactive, tasactive
    
    
    def reso7(self, idxown): 
        'The descend into resolution layer + hover strategy'
        
        # TODO: determine the correct altitude for the drone to descend to  (maybe call reso1)
        newalt = traf.alt[idxown]
        altactive = True
        
        # TODO: make it hover
        newgs = traf.gs[idxown] # 0.0
        tasactive = True
                
        return newalt, newgs, altactive, tasactive
    