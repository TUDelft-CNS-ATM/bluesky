"""A plugin for playing air traffic from the OpenSky Network.

The plugin gets current traffic from the OpenSky Network and makes aircraft
move.

The OpenSky Python API allows for a number of unauthenticated requests. If you
feed the network, you are allowed an unlimited number of requests for the data
you feed. If you set `opensky_user` and `opensky_password` in your settings.cfg
file and the latest request you send fails, the program falls back to the data
from your sensors.

Xavier Olive, 2018
Joost Ellerbroek, 2018
"""
import time
import pickle
import requests
import numpy as np

from bluesky import stack, settings, traf
from bluesky.core import Entity, timed_function
from bluesky.tools import cachefile
settings.set_variable_defaults(opensky_user=None, opensky_password=None,
                               opensky_ownonly=False)

# Globals
actypedb_version = 'v20180126'
reader = None
actypes = dict()

def init_plugin():
    global reader
    reader = OpenSkyListener()

    # Load aircraft type database
    get_actypedb()

    config = {
        'plugin_name': 'OPENSKY',
        'plugin_type': 'sim'
    }

    return config

def get_actypedb():
    ''' Get aircraft type database from cache or web. '''
    global actypes
    with cachefile.openfile('actypedb.p', actypedb_version) as cache:
        try:
            actypes = cache.load()
        except (pickle.PickleError, cachefile.CacheError) as e:
            print(e.args[0])
            import io, zipfile
            f = requests.get('https://junzisun.com/adb/download')
            with zipfile.ZipFile(io.BytesIO(f.content)) as zfile:
                with zfile.open('aircraft_db.csv', 'r') as dbfile:
                    actypes = dict([line.decode().split(',')[0:3:2] for line in dbfile])

            cache.dump(actypes)


class OpenSkyListener(Entity):
    def __init__(self):
        super().__init__()
        if settings.opensky_user:
            self._auth = (settings.opensky_user, settings.opensky_password)
        else:
            self._auth = ()
        self._api_url = "https://opensky-network.org/api"
        self.connected = False

        with self.settrafarrays():
            self.upd_time = np.array([])
            self.my_ac = np.array([], dtype=bool)

    def create(self, n=1):
        super().create(n)
        # Store creation time of new aircraft
        self.upd_time[-n:] = time.time()
        self.my_ac[-n:] = False

    def get_json(self, url_post, params=None):
        r = requests.get(self._api_url + url_post, auth=self._auth, params=params)
        if r.status_code == 200:
            return r.json()

        # "Response not OK. Status {0:d} - {1:s}".format(r.status_code, r.reason)
        return None

    def get_states(self, ownonly=False):
        url_post = '/states/{}'.format('own' if ownonly else 'all')
        states_json = self.get_json(url_post)
        if states_json is not None:
            return list(zip(*states_json['states']))
        return None

    @timed_function(name='OPENSKY', dt=6.0)
    def update(self):
        if not self.connected:
            return

        # t1 = time.time()
        if settings.opensky_ownonly:
            states = self.get_states(ownonly=True)
            if states is None:
                return
        else:
        # Get states from OpenSky. If all states fails try getting own states only.
            states = self.get_states()
            if states is None:
                if self.authenticated:
                    states = self.get_states(ownonly=True)
                if states is None:
                    return

        # Current time
        curtime = time.time()

        # States contents:
        icao24, acid, orig, time_pos, last_contact, lon, lat, geo_alt, on_gnd, \
            spd, hdg, vspd, sensors, baro_alt, squawk, spi, pos_src = states[:17]

        # Relevant params as numpy arrays
        lat = np.array(lat, dtype=np.float64)
        lon = np.array(lon, dtype=np.float64)
        alt = np.array(baro_alt, dtype=np.float64)
        hdg = np.array(hdg, dtype=np.float64)
        vspd = np.array(vspd, dtype=np.float64)
        spd = np.array(spd, dtype=np.float64)
        acid = np.array([i.strip() for i in acid], dtype=np.str_)
        icao24 = np.array(icao24, dtype=np.str_)

        idx = np.array(traf.id2idx(acid))

        # Split between already existing and new aircraft
        newac = idx == -1
        other = np.logical_not(newac)

        # Filter out invalid entries
        valid = np.logical_not(np.logical_or.reduce(
            [np.isnan(x) for x in [lat, lon, alt, hdg, vspd, spd]]))
        newac = np.logical_and(newac, valid)
        other = np.logical_and(other, valid)
        n_new = np.count_nonzero(newac)
        n_oth = np.count_nonzero(other)

        # t2 = time.time()

        # Create new aircraft
        if n_new:
            actype = [actypes.get(str(i), 'B744') for i in icao24[newac]]
            for j in range(n_new):
                traf.cre(n_new, acid[newac][j], actype[j], lat[newac][j], lon[newac][j],\
                         hdg[newac][j], alt[newac][j], spd[newac][j])
            self.my_ac[-n_new:] = True

        # t3 = time.time()

        # Update the rest
        if n_oth:
            traf.move(idx[other], lat[other], lon[other], alt[other], hdg[other], \
                      spd[other], vspd[other])
            self.upd_time[idx[other]] = curtime

        # t4 = time.time()

        # remove aircraft with no message for less than 1 minute
        # opensky already filters
        delidx = np.where(np.logical_and(self.my_ac, curtime - self.upd_time > 10))[0]
        if len(delidx) > 0:
            traf.delete(delidx)

        # t5 = time.time()
        # print('req={}, mod={}, cre={}, mov={}, del={}, ncre={}, nmov={}, ndel={}'.format(curtime-t1, t2-curtime, t3-t2, t4-t3, t5-t4, n_new, n_oth, len(delidx)))

    @stack.command(name='OPENSKY')
    def toggle(self, flag:bool=None):
        ''' Use the OpenSky plugin as a data source for traffic. '''
        if flag is None:
            return True, f'OpenSky is currently {"" if self.connected else "not"} connected.'
        self.connected = flag
        if flag:
            stack.stack('OP')
            return True, 'Connecting to OpenSky'
        return True, 'Stopping the requests'
