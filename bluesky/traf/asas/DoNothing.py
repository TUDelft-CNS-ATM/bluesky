# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 14:27:44 2015

@author: Jerom Maas
"""

def start(dbconf):
    pass

def resolve(dbconf, traf):
    
    # If resolution is off, and detection is on, and a conflict is detected
    # then asas will be active for that airplane. Since resolution is off, it
    # should then follow the auto pilot instructions.   
    dbconf.trk = traf.ap.trk
    dbconf.spd = traf.ap.tas
    dbconf.vs  = traf.ap.vs
    dbconf.alt = traf.ap.alt
    
    return
