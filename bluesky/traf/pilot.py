import numpy as np
import bluesky as bs
from ..tools.aero import vtas2eas, vcas2tas, vcas2mach
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters


class Pilot(DynamicArrays):
    def __init__(self):
        with RegisterElementParameters(self):
            # Desired aircraft states
            self.alt = np.array([])  # desired altitude [m]
            self.hdg = np.array([])  # desired heading [deg]
            self.trk = np.array([])  # desired track angle [deg]
            self.vs  = np.array([])  # desired vertical speed [m/s]
            self.spd = np.array([])  # desired speed [m/s]

    def create(self, n=1):
        super(Pilot, self).create(n)

        self.alt[-n:] = bs.traf.alt[-n:]
        self.spd[-n:] = vtas2eas(bs.traf.tas[-n:], bs.traf.alt[-n:])
        self.hdg[-n:] = bs.traf.hdg[-n:]
        self.trk[-n:] = bs.traf.trk[-n:]

    def FMSOrAsas(self):
        #--------- Input to Autopilot settings to follow: destination or ASAS ----------
        # Convert the ASAS commanded speed from ground speed to TAS
        if bs.traf.wind.winddim > 0:
            vwn, vwe     = bs.traf.wind.getdata(bs.traf.lat, bs.traf.lon, bs.traf.alt)
            asastasnorth = bs.traf.asas.spd * np.cos(np.radians(bs.traf.asas.trk)) - vwn
            asastaseast  = bs.traf.asas.spd * np.sin(np.radians(bs.traf.asas.trk)) - vwe
            asastas      = np.sqrt(asastasnorth**2 + asastaseast**2)
        # no wind, then ground speed = TAS
        else:
            asastas = bs.traf.asas.spd

        # Determine desired states from ASAS or AP. Select asas if there is a conflict AND resolution is on.
        self.trk = np.where(bs.traf.asas.active, bs.traf.asas.trk, bs.traf.ap.trk)
        self.spd = np.where(bs.traf.asas.active, asastas, bs.traf.ap.tas)
        self.alt = np.where(bs.traf.asas.active, bs.traf.asas.alt, bs.traf.ap.alt)
        self.vs  = np.where(bs.traf.asas.active, bs.traf.asas.vs, bs.traf.ap.vs)

        # ASAS can give positive and negative VS, but the sign of VS is determined using delalt in Traf.ComputeAirSpeed
        # Therefore, ensure that pilot.vs is always positive to prevent opposite signs of delalt and VS in Traf.ComputeAirSpeed
        self.vs = np.abs(self.vs)

        # Compute the desired heading needed to compensate for the wind
        if bs.traf.wind.winddim > 0:

            # Calculate wind correction
            vwn, vwe = bs.traf.wind.getdata(bs.traf.lat, bs.traf.lon, bs.traf.alt)
            Vw       = np.sqrt(vwn * vwn + vwe * vwe)
            winddir  = np.arctan2(vwe, vwn)
            drift    = np.radians(self.trk) - winddir  # [rad]
            steer    = np.arcsin(np.minimum(1.0, np.maximum(-1.0,
                                     Vw * np.sin(drift) / np.maximum(0.001, bs.traf.tas))))
            # desired heading
            self.hdg = (self.trk + np.degrees(steer)) % 360.
        else:
            self.hdg = self.trk % 360.

    def FlightEnvelope(self):
        # check for the flight envelope
        bs.traf.delalt = bs.traf.apalt - bs.traf.alt  # [m]
        bs.traf.perf.limits()

        #print self.aspd[0]/kts

        # Update desired sates with values within the flight envelope
        # To do: add const Mach const CAS mode

        self.spd = np.where(bs.traf.limspd_flag, vcas2tas(bs.traf.limspd, bs.traf.alt), self.spd)

        # Autopilot selected altitude [m]
        self.alt = np.where(bs.traf.limalt > -900., bs.traf.limalt, self.alt)

        # Autopilot selected vertical speed (V/S)
        self.vs = np.where(bs.traf.limvs > -9000., bs.traf.limvs, self.vs)

        # To be discussed: Following change in VNAV mode only?
        # below crossover altitude: CAS=const, above crossover altitude: MA = const
        # climb/descend above crossover: Ma = const, else CAS = const
        # ama is fixed when above crossover
        bs.traf.ama = np.where(bs.traf.abco * (bs.traf.ama == 0.),
                                 vcas2mach(bs.traf.aspd, bs.traf.alt), bs.traf.ama)

        # ama is deleted when below crossover
        bs.traf.ama = np.where(bs.traf.belco, 0.0, bs.traf.ama)
