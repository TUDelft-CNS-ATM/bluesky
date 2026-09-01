""" BlueSky kinematics implementation.

This module contains the point-mass kinematics that integrate the aircraft
state (airspeed, heading, vertical speed, ground speed, track and position)
each simulation step. It is factored out of Traffic into a separate,
replaceable Entity so that the kinematics model can be swapped for an
alternative implementation (e.g. from a plugin) with Kinematics.select().

The fundamental aircraft state (lat, lon, alt, tas, hdg, vs, ...) lives on
Traffic and is shared with the rest of BlueSky; the kinematics model reads and
writes it through the global bs.traf reference, exactly like the other flight
models (autopilot, ASAS, ...). The state produced by the integration itself
(the longitudinal/vertical accelerations and the turn/altitude-capture
switches) is owned here as traffic arrays.
"""
import numpy as np

import bluesky as bs
from bluesky.core import Entity
from bluesky.tools.aero import fpm, ft, g0, Rearth, vtas2cas, vtas2mach


class Kinematics(Entity, replaceable=True):
    """ Default BlueSky point-mass kinematics.

    Integrates airspeed/heading/vertical speed, derives ground speed and
    track (including wind), and updates the aircraft position each step.
    """

    def __init__(self):
        super().__init__()

        with self.settrafarrays():
            # Accelerations produced by the integration
            self.ax       = np.array([])              # [m/s2] current longitudinal acceleration
            self.az       = np.array([])              # [m/s2] current vertical acceleration

            # Guidance-mode switches produced by the integration
            self.swhdgsel = np.array([], dtype=bool)  # whether aircraft is turning
            self.swaltsel = np.array([], dtype=bool)  # whether altitude capture/hold is engaged

    def create(self, n=1):
        super().create(n)

        self.ax[-n:]       = 0.0
        self.az[-n:]       = 0.0
        self.swhdgsel[-n:] = False
        self.swaltsel[-n:] = False

    def update(self):
        """ Perform one kinematic integration step for all aircraft. """
        self.update_airspeed()
        self.update_groundspeed()
        self.update_pos()

    def update_airspeed(self):
        traf = bs.traf
        # Compute horizontal acceleration
        delta_spd = traf.aporasas.tas - traf.tas
        need_ax = np.abs(delta_spd) > np.abs(bs.sim.simdt * traf.perf.axmax)
        self.ax = need_ax * np.sign(delta_spd) * traf.perf.axmax
        # Update velocities
        traf.tas = np.where(need_ax, traf.tas + self.ax * bs.sim.simdt, traf.aporasas.tas)
        traf.cas = vtas2cas(traf.tas, traf.alt)
        traf.M = vtas2mach(traf.tas, traf.alt)

        # Turning bank triangle
        # tan phi = a centrigugal/a grav = omega^2 * R / g = omega * V /g
        # => omega = (g tan phi)/V
        turnrate = np.degrees(g0 * np.tan(np.where(traf.ap.turnphi > traf.eps * traf.eps,
                                                   traf.ap.turnphi, traf.ap.bankdef) )
                                          / np.maximum(traf.tas, traf.eps))
        delhdg = (traf.aporasas.hdg - traf.hdg + 180) % 360 - 180  # [deg]
        self.swhdgsel = np.abs(delhdg) > np.abs(bs.sim.simdt * turnrate)

        # Update heading
        traf.hdg = np.where(self.swhdgsel,
                            traf.hdg + bs.sim.simdt * turnrate * np.sign(delhdg), traf.aporasas.hdg) % 360.0

        # Update vertical speed (alt select, capture and hold autopilot mode)
        delta_alt = traf.aporasas.alt - traf.alt
        # Old dead band version:
        #        self.swaltsel = np.abs(delta_alt) > np.maximum(
        #            10 * ft, np.abs(2 * bs.sim.simdt * traf.vs))

        # Update version: time based engage of altitude capture (to adapt for UAV vs airliner scale)
        self.swaltsel = np.abs(delta_alt) > 1.05 * np.maximum(np.abs(bs.sim.simdt * traf.aporasas.vs),
                                                         np.abs(bs.sim.simdt * traf.vs))
        target_vs = self.swaltsel * np.sign(delta_alt) * np.abs(traf.aporasas.vs)
        delta_vs = target_vs - traf.vs
        # print(delta_vs / fpm)
        need_az = np.abs(delta_vs) > 300 * fpm   # small threshold
        self.az = need_az * np.sign(delta_vs) * (300 * fpm)   # fixed vertical acc approx 1.6 m/s^2
        traf.vs = np.where(need_az, traf.vs + self.az * bs.sim.simdt, target_vs)
        traf.vs = np.where(np.isfinite(traf.vs), traf.vs, 0)    # fix vs nan issue

    def update_groundspeed(self):
        traf = bs.traf
        # Compute ground speed and track from heading, airspeed and wind
        if traf.wind.winddim == 0:  # no wind
            traf.gsnorth  = traf.tas * np.cos(np.radians(traf.hdg))
            traf.gseast   = traf.tas * np.sin(np.radians(traf.hdg))

            traf.gs  = traf.tas
            traf.trk = traf.hdg
            traf.windnorth[:], traf.windeast[:] = 0.0, 0.0

        else:
            applywind = traf.alt > 50. * ft  # Only apply wind when airborne

            vnwnd, vewnd = traf.wind.getdata(traf.lat, traf.lon, traf.alt)
            traf.windnorth[:], traf.windeast[:] = vnwnd, vewnd
            traf.gsnorth  = traf.tas * np.cos(np.radians(traf.hdg)) + traf.windnorth * applywind
            traf.gseast   = traf.tas * np.sin(np.radians(traf.hdg)) + traf.windeast * applywind

            traf.gs  = np.logical_not(applywind) * traf.tas + \
                       applywind * np.sqrt(traf.gsnorth**2 + traf.gseast**2)

            traf.trk = np.logical_not(applywind) * traf.hdg + \
                       applywind * np.degrees(np.arctan2(traf.gseast, traf.gsnorth)) % 360.

        traf.work += (traf.perf.thrust * bs.sim.simdt * np.sqrt(traf.gs * traf.gs + traf.vs * traf.vs))

    def update_pos(self):
        traf = bs.traf
        # Update position
        traf.alt = np.where(self.swaltsel, np.round(traf.alt + traf.vs * bs.sim.simdt, 6), traf.aporasas.alt)
        traf.lat = traf.lat + np.degrees(bs.sim.simdt * traf.gsnorth / Rearth)
        traf.coslat = np.cos(np.deg2rad(traf.lat))
        traf.lon = traf.lon + np.degrees(bs.sim.simdt * traf.gseast / traf.coslat / Rearth)
        traf.distflown += traf.gs * bs.sim.simdt
