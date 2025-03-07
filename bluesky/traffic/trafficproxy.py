import numpy as np

from bluesky.core import Base
from bluesky.network.sharedstate import ActData


class TrafficProxy(Base):
    ''' Proxy object implementation for Traffic()-like access
        of received data on client side.
    '''
    id: ActData[list] = ActData(group='acdata')
    lat: ActData[np.ndarray] = ActData(0, group='acdata')
    lon: ActData[np.ndarray] = ActData(0, group='acdata')
    alt: ActData[np.ndarray] = ActData(0, group='acdata')
    trk: ActData[np.ndarray] = ActData(0, group='acdata')
    cas: ActData[np.ndarray] = ActData(0, group='acdata')
    tas: ActData[np.ndarray] = ActData(0, group='acdata')
    gs: ActData[np.ndarray] = ActData(0, group='acdata')
    vs: ActData[np.ndarray] = ActData(0, group='acdata')
    translvl: ActData[float] = ActData(0.0, group='acdata')

    # data['ingroup']    = bs.traf.groups.ingroup
    # data['inconf'] = bs.traf.cd.inconf
    # data['tcpamax'] = bs.traf.cd.tcpamax
    # data['rpz'] = bs.traf.cd.rpz
    # data['nconf_cur'] = len(bs.traf.cd.confpairs_unique)
    # data['nconf_tot'] = len(bs.traf.cd.confpairs_all)
    # data['nlos_cur'] = len(bs.traf.cd.lospairs_unique)
    # data['nlos_tot'] = len(bs.traf.cd.lospairs_all)
    
    # data['vmin']       = bs.traf.perf.vmin
    # data['vmax']       = bs.traf.perf.vmax


    # # Send casmachthr for route visualization
    # data['casmachthr']    = aero.casmach_thr

    # ASAS resolutions for visualization. Only send when evaluated
    # data['asastas']  = bs.traf.cr.tas
    # data['asastrk']  = bs.traf.cr.trk

    # # Aircraft (group) color
    # if self.custacclr:
    #     data['custacclr'] = self.custacclr
    # if self.custgrclr:
    #     data['custgrclr'] = self.custgrclr

    @property
    def ntraf(self):
        return len(self.id)

    def id2idx(self, acid):
        """Find index of aircraft id"""
        if not isinstance(acid, str):
            # id2idx is called for multiple id's
            # Fast way of finding indices of all ACID's in a given list
            tmp = dict((v, i) for i, v in enumerate(self.id))
            return [tmp.get(acidi, -1) for acidi in acid]
        else:
             # Catch last created id (* or # symbol)
            if acid in ('#', '*'):
                return self.ntraf - 1

            try:
                return self.id.index(acid.upper())
            except:
                return -1
