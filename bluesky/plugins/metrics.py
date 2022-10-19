""" Airspace metrics plugin. """
import numpy as np
from bluesky import stack, traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.core import Entity, timed_function
from bluesky.tools import areafilter, datalog, plotter, geo
from bluesky.tools.aero import nm, ft

# Metrics object
metrics = None

class SectorData:
    # Selected traffic data for a/c in a sector
    def __init__(self):
        self.acid = list()
        self.lat0 = np.array([])
        self.lon0 = np.array([])
        self.dist0 = np.array([])

    def id2idx(self, acid):
        # Fast way of finding indices of all ACID's in our list
        tmp = dict((v, i) for i, v in enumerate(self.acid))
        # Return only the indices
        return np.array([tmp.get(acidi, -1) for acidi in acid], dtype=int)

    def get(self, acid):
        # Get lat,lon and distance flown for this a/c
        idx = self.id2idx(acid)
        return self.lat0[idx], self.lon0[idx], self.dist0[idx]

    def delete(self, acid):
        # Remove an aircraft from our sector traffic data list
        idx = np.sort(self.id2idx(acid))
        self.lat0 = np.delete(self.lat0, idx)
        self.lon0 = np.delete(self.lon0, idx)
        self.dist0 = np.delete(self.dist0, idx)
        for i in reversed(idx):
            del self.acid[i]

    def extend(self, acid, lat0, lon0, dist0):
        # Add several a/c to our list:
        # input: lat0,lon0,dist: np arrays, acid: list of strings
        self.lat0 = np.append(self.lat0, lat0)
        self.lon0 = np.append(self.lon0, lon0)
        self.dist0 = np.append(self.dist0, dist0)
        self.acid.extend(acid)

class Metrics(Entity):
    def __init__(self):
        super().__init__()
        # List of sectors known to this plugin.
        self.sectors = list()
        # List of sets of aircraft in each sector
        self.acinside = list()
        # Static Density metric
        self.sectorsd = np.array([], dtype=np.int)
        # Summed pairwise convergence metric
        self.sectorconv = np.array([], dtype=np.float)
        # Route efficiency metric
        self.sectoreff = []

        self.effplot = None

        self.delac = SectorData()

        self.fsd = None
        self.fconv = None
        self.feff = None

    def create(self, n=1):
        pass
        # print(n, 'aircraft created, ntraf =', traf.ntraf)

    def delete(self, idx):
        self.delac.extend(np.array(traf.id)[idx], traf.lat[idx], traf.lon[idx], traf.distflown[idx])
        # n = len(idx) if isinstance(idx, Collection) else 1
        # print(n, 'aircraft deleted, ntraf =', traf.ntraf, 'idx =', idx, 'len(traf.lat) =', len(traf.lat))

    @timed_function(dt=2.5)
    def update(self):
        ''' Periodic update function for metrics calculation. '''
        self.sectorsd = np.zeros(len(self.sectors))
        self.sectorconv = np.zeros(len(self.sectors))
        self.sectoreff = []
        if not traf.ntraf or not self.sectors:
            return

        # Check convergence using CD with large RPZ and tlook
        confpairs, lospairs, inconf, tcpamax, qdr, dist, dcpa, tcpa, tLOS = \
            traf.cd.detect(traf, traf, np.ones(traf.ntraf) * 20 * nm, traf.cd.hpz, np.ones(traf.ntraf) * 3600)

        if confpairs:
            own, intr = zip(*confpairs)
            ownidx = traf.id2idx(own)
            mask = traf.alt[ownidx] > 70 * ft
            ownidx = np.array(ownidx)[mask]
            dcpa = np.array(dcpa)[mask]
            tcpa = np.array(tcpa)[mask]
        else:
            ownidx = np.array([])
    
        sendeff = False
        for idx, (sector, previnside) in enumerate(zip(self.sectors, self.acinside)):
            inside = areafilter.checkInside(sector, traf.lat, traf.lon, traf.alt)

            sectoreff = []
            # Detect aircraft leaving and entering the sector
            previds = set(previnside.acid)
            ids = set(np.array(traf.id)[inside])
            arrived = list(ids - previds)
            left = previds - ids

            # Split aircraft that left the sector in deleted and not deleted
            left_intraf = left.intersection(traf.id)
            left_del = list(left - left_intraf) # Aircraft id's prev inside but deleted
            left_intraf = list(left_intraf) # Aircraft id's that left sector

            # New a/c in sector arr listed by index in arridx
            arridx = traf.id2idx(arrived)
            leftidx = traf.id2idx(left_intraf)

            # Retrieve the current distance flown for arriving and leaving aircraft
            arrdist = traf.distflown[arridx]
            arrlat = traf.lat[arridx]
            arrlon = traf.lon[arridx]

            # Get all a/c ids that left from the set delac
            leftlat, leftlon, leftdist = self.delac.get(left_del)
            leftlat = np.append(leftlat, traf.lat[leftidx])
            leftlon = np.append(leftlon, traf.lon[leftidx])
            leftdist = np.append(leftdist, traf.distflown[leftidx])
            leftlat0, leftlon0, leftdist0 = previnside.get(left_del + left_intraf)
            self.delac.delete(left_del)

            if len(left) > 0:

                # Exclude aircraft where origin = destination for sector efficiency,
                # so require that distance start-end > 10 nm
                q, d = geo.qdrdist(leftlat0, leftlon0, leftlat, leftlon)
                mask = d > 10

                sectoreff = list((leftdist[mask] - leftdist0[mask]) / d[mask] / nm)
                names = np.array(left_del + left_intraf)[mask]

                for name, eff in zip(names, sectoreff):
                    self.feff.write(f'{sim.simt}, {name}, {eff}\n')
                sendeff = True

                # print('{} aircraft left sector {}, distance flown (acid:dist):'.format(len(left), sector))
                # for a, d0, d1, e in zip(left, leftdist0, leftdist, sectoreff):
                #     print('Aircraft {} flew {} meters (eff = {})'.format(a, round(d1-d0), e))

            # Update inside data for this sector
            previnside.delete(left)
            previnside.extend(arrived, arrlat, arrlon, arrdist)

            self.sectoreff.append(sectoreff)

            self.sectorsd[idx] = np.count_nonzero(inside)
            insidx = np.where(np.logical_and(inside, inconf))
            pairsinside = np.isin(ownidx, insidx)
            if len(pairsinside):
                tnorm = np.array(tcpa)[pairsinside] / 300.0
                dcpanorm = np.array(dcpa)[pairsinside] / (5.0 * nm)
                self.sectorconv[idx] = np.sum(
                    np.sqrt(2.0 / tnorm * tnorm + dcpanorm * dcpanorm))
            else:
                self.sectorconv[idx] = 0

            self.fconv.write(f'{sim.simt}, {self.sectorconv[idx]}\n')
            self.fsd.write('{sim.simt}, {self.sectorsd[idx]}\n')
        if sendeff:
            self.effplot.send()

    def reset(self):
        if self.fconv:
            self.fconv.close()
        if self.fsd:
            self.fsd.close()
        if self.feff:
            self.feff.close()

    @stack.command(name='METRICS')
    def stackio(self, cmd:'txt', name:'txt'=''):
        ''' Calculate a set of metrics within specified sectors. '''
        print('BLAAA', cmd, name)
        if cmd == 'LIST':
            if not self.sectors:
                return True, 'No registered sectors available'
            return True, 'Registered sectors:', str.join(', ', self.sectors)
        elif cmd == 'ADDSECTOR':
            if name == 'ALL':
                for areaname in areafilter.basic_shapes.keys():
                    self.stackio('ADDSECTOR', areaname)
            # Add new sector to list.
            elif name in self.sectors:
                return False, 'Sector %s already registered.' % name
            elif areafilter.hasArea(name):
                if not self.sectors:
                    self.fconv = open('output/'+stack.get_scenname()+'convergence.csv', 'w')
                    self.fsd = open('output/'+stack.get_scenname()+'density.csv', 'w')
                    self.feff = open('output/'+stack.get_scenname()+'efficiency.csv', 'w')
                    # Create the plot if this is the first sector
                    plotter.plot('metrics.metrics.sectorsd', dt=2.5, title='Static Density',
                                xlabel='Time', ylabel='Aircraft count', fig=1)
                    plotter.plot('metrics.metrics.sectorconv', dt=2.5, title='Summed Pairwise Convergence',
                                xlabel='Time', ylabel='Convergence', fig=2)
                    self.effplot = plotter.Plot('metrics.metrics.sectoreff', title='Route Efficiency', plot_type='boxplot',
                                xlabel='Sector', ylabel='Efficiency', fig=3)
                # Add new area to the sector list, and add an initial inside count of traffic
                self.sectors.append(name)
                self.acinside.append(SectorData())
                plotter.legend(self.sectors, 1)
                return True, 'Added %s to sector list.' % name

            else:
                return False, "No area found with name '%s', create it first with one of the shape commands" % name

        else:
            # Remove area from sector list
            if name in self.sectors:
                idx = self.sectors.index(name)
                self.sectors.pop(idx)
                return True, 'Removed %s from sector list.' % name
            return False, "No sector registered with name '%s'." % name

def init_plugin():
    # Addtional initilisation code
    global metrics
    metrics = Metrics()
    # Configuration parameters
    config = {
        'plugin_name': 'METRICS',
        'plugin_type': 'sim'
        }

    return config
