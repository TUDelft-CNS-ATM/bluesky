import numpy as np
from bluesky.tools.aero import nm, ft
from bluesky.traffic.asas import MVP
from bluesky import traf
# TODO: not completely migrated yet to class-based implementation


def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SWARM',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim'
    }

    return config


class Swarm(MVP):
    def __init__(self):
        super().__init__()
        self.rpzswarm = 7.5 * nm  # [m]
        self.dhswarm = 1500 * ft  # [m]
        self.Swarmweights = np.array([10, 3, 1])

    def resolve(self, conf, ownship, intruder):
        # Find matrix of neighbouring aircraft withing swarm distance
        qdrrad = np.radians(conf.qdr)
        dx = conf.dist * np.sin(qdrrad)
        dy = conf.dist * np.cos(qdrrad)

        dy = dy - np.eye(ownship.ntraf) * 1e9  # setting distance of A/C to itself to 0,\
        # correcting the distance from CASAS line 109

        dalt = ownship.alt.reshape((1, ownship.ntraf)) - ownship.alt.reshape((1, ownship.ntraf)).T
        close = np.logical_and(dx**2 + dy**2 < conf.rpzswarm**2,
                            np.abs(dalt) < self.dhswarm)

        trkdif = ownship.trk.reshape(1, ownship.ntraf) - ownship.trk.reshape(ownship.ntraf, 1)
        dtrk = (trkdif + 180) % 360 - 180
        samedirection = np.abs(dtrk) < 90

        selected = np.logical_and(close, samedirection)
        own = np.eye(ownship.ntraf, dtype='bool')

        Swarming = np.logical_or(selected, own)

        # First do conflict resolution following MVP
        newtrk, newgs, newvs, newalt = super().resolve(conf, ownship, intruder)

        # Find desired speed vector after Collision Avoidance or Autopilot
        ca_trk = np.where(conf.inconf, newtrk, ownship.ap.trk)
        ca_cas = np.where(conf.inconf, newgs, ownship.selspd)
        ca_vs = np.where(conf.inconf, newvs, ownship.selvs)

        # Add factor of Velocity Alignment to speed vector
        hspeed = np.ones((ownship.ntraf, ownship.ntraf)) * ownship.cas
        va_cas = np.average(hspeed, axis=1, weights=Swarming)

        vspeed = np.ones((ownship.ntraf, ownship.ntraf)) * ownship.vs
        va_vs = np.average(vspeed, axis=1, weights=Swarming)

        avgdtrk = np.average(dtrk, axis=1, weights=Swarming)
        va_trk = ownship.trk + avgdtrk

        # Add factor of Flock Centering to speed vector
        dxflock = dx + np.eye(ownship.ntraf) * asas.u / 100.
        dyflock = dy + np.eye(ownship.ntraf) * asas.v / 100.

        fc_dx = np.average(dxflock, axis=1, weights=Swarming)
        fc_dy = np.average(dyflock, axis=1, weights=Swarming)

        z = np.ones((traf.ntraf, traf.ntraf)) * traf.alt
        fc_dz = np.average(z, axis=1, weights=Swarming) - traf.alt

        fc_trk = np.degrees(np.arctan2(fc_dx, fc_dy))
        fc_cas = traf.cas
        ttoreach = np.sqrt(fc_dx**2 + fc_dy**2) / fc_cas
        fc_vs = np.where(ttoreach == 0, 0, fc_dz / ttoreach)

        # Find final Swarming directions
        trks = np.array([ca_trk, va_trk, fc_trk])
        cass = np.array([ca_cas, va_cas, fc_cas])
        vss = np.array([ca_vs, va_vs, fc_vs])

        trksrad = np.radians(trks)
        vxs = cass * np.sin(trksrad)
        vys = cass * np.cos(trksrad)

        Swarmvx = np.average(vxs, axis=0, weights=asas.Swarmweights)
        Swarmvy = np.average(vys, axis=0, weights=asas.Swarmweights)
        Swarmhdg = np.degrees(np.arctan2(Swarmvx, Swarmvy))
        Swarmcas = np.average(cass, axis=0, weights=asas.Swarmweights)
        Swarmvs = np.average(vss, axis=0, weights=asas.Swarmweights)

        # Cap the velocity
        Swarmcascapped = np.maximum(traf.perf.vmin, np.minimum(traf.perf.vmax, Swarmcas))
        # Assign Final Swarming directions to traffic
        asas.hdg = Swarmhdg
        asas.tas = Swarmcascapped
        asas.vs = Swarmvs
        asas.alt = np.sign(Swarmvs) * 1e5

        # Make sure that all aircraft follow these directions
        asas.active.fill(True)
