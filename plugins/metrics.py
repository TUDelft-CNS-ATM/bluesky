""" Airspace metrics plugin. """
import numpy as np

from bluesky import stack, traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import areafilter, datalog, plotter
from bluesky.tools.aero import nm
# List of sectors known to this plugin.
sectors = list()

# Static Density metric
sectorsd = np.array([], dtype=np.int)
# Summed pairwise convergence metric
sectorconv = np.array([], dtype=np.float)


def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        'plugin_name': 'METRICS',
        'plugin_type': 'sim',
        'update_interval': 2.5,
        'update': update,
        'reset': reset
        }

    stackfunctions = {
        'METRICS': [
            'METRICS ADDSECTOR name',
            'txt,txt',
            stackio,
            'Print something to the bluesky console based on the flag passed to MYFUN.']
    }

    return config, stackfunctions


def update():
    global sectorsd, sectorconv
    if not traf.ntraf or not sectors:
        sectorsd = np.zeros(np.size(sectorsd))
        sectorconv = np.zeros(np.size(sectorconv))
        return

    # Check convergence using CD with large RPZ and tlook
    confpairs, lospairs, inconf, tcpamax, qdr, dist, dcpa, tcpa, tLOS = \
        traf.asas.cd.detect(traf, traf, 50 * nm, traf.asas.dh, 3600)

    if confpairs:
        own, int = zip(*confpairs)
        ownidx = traf.id2idx(own)
    else:
        ownidx = np.array([])

    sectorsd = np.zeros(len(sectors))
    sectorconv = np.zeros(len(sectors))
    for idx, sector in enumerate(sectors):
        inside = areafilter.checkInside(sector, traf.lat, traf.lon, traf.alt)
        sectorsd[idx] = np.count_nonzero(inside)

        insidx = np.where(inside & inconf)
        pairsinside = np.isin(ownidx, insidx)

        tnorm = tcpa[pairsinside] / 300.0
        dcpanorm = dcpa[pairsinside] / (5.0 * nm)
        sectorconv[idx] = np.sum(1.0 / np.sqrt(tnorm * tnorm + dcpanorm * dcpanorm))

def reset():
    pass

def stackio(cmd, name):
    if cmd == 'LIST':
        if len(sectors) == 0:
            return True, 'No registered sectors available'
        else:
            return True, 'Registered sectors:', str.join(', ', sectors)
    elif cmd == 'ADDSECTOR':
        # Add new sector to list.
        if name in sectors:
            return True, 'Sector %s already registered.' % name
        elif areafilter.hasArea(name):
            if not sectors:
                # Create the plot if this is the first sector
                plotter.plot('metrics.sectorsd', dt=2.5, title='Static Density', xlabel='Time', ylabel='Aircraft count', fig=1)
                plotter.plot('metrics.sectorconv', dt=2.5, title='Summed Pairwise Convergence', xlabel='Time', ylabel='Convergence', fig=2)
            # Add new area to the sector list, and add an initial inside count of traffic
            sectors.append(name)
            plotter.legend(sectors, 1)
            return True, 'Added %s to sector list.' % name
        else:
            return False, "No area found with name '%s', create it first with one of the shape commands" % name

    else:
        # Remove area from sector list
        if name in sectors:
            idx = sectors.index(name)
            sectors.pop(idx)
            return True, 'Removed %s from sector list.' % name
        else:
            return False, "No sector registered with name '%s'." % name
