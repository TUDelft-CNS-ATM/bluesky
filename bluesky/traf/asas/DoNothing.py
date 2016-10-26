# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 14:27:44 2015

@author: Jerom Maas
"""

def start(dbconf):
    pass

def resolve(dbconf, traf):
    
    # When resolution is OFF, and aircraft is in conflict, then asas will be active
    # Under these conditions, the aircraft should listen to AP
    dbconf.trk = traf.ap.trk
    dbconf.spd = traf.ap.tas
    dbconf.vs  = traf.ap.vs
    dbconf.alt = traf.ap.alt
    
    return
