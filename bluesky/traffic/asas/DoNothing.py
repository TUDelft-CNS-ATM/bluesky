# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 14:27:44 2015

@author: Jerom Maas
"""

def start(asas):
    pass

def resolve(asas, traf):
    
    # If resolution is off, and detection is on, and a conflict is detected
    # then asas will be active for that airplane. Since resolution is off, it
    # should then follow the auto pilot instructions.   
    asas.trk = traf.ap.trk[:]
    asas.tas = traf.ap.tas[:]
    asas.vs  = traf.ap.vs[:]
    asas.alt = traf.ap.alt[:]
    
    return
