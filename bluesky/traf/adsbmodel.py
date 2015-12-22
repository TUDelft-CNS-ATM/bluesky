import numpy as np
from math import *

from ..tools.aero import fpm, kts, ft, nm, g0,  tas2eas, tas2mach, tas2cas, mach2cas,  \
     cas2tas, temp, density, Rearth

from ..tools.aero_np import vatmos, vcas2tas, vtas2cas,  vtas2mach, vcas2mach, \
    vmach2tas, qdrdist
from ..tools.misc import degto180
from ..tools.datalog import Datalog

class ADSBModel():
    """
    Traffic class definition    : Traffic data

    Methods:
        Traffic()            :  constructor

        create(acid,actype,aclat,aclon,achdg,acalt,acspd) : create aircraft
        delete(acid)         : delete an aircraft from traffic data
        update(sim)          : do a numerical integration step
        trafperf ()          : calculate aircraft performance parameters

    Members: see create

    Created by  : Jacco M. Hoekstra
    """

    def __init__(self, traf):
        
        self.traf = traf        
        
    def update(self):
        return
        
 