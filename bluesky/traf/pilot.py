import numpy as np
from ..tools.aero import tas2eas, vcas2tas, vcas2mach
from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters


class Pilot(DynamicArrays):
    def __init__(self, traf):
        self.traf = traf
        self.ap  = traf.ap
        self.asas = self.traf.asas

        with RegisterElementParameters(self):
            # Desired aircraft states
            self.alt = np.array([])  # desired altitude [m]
            self.hdg = np.array([])  # desired heading [deg]
            self.trk = np.array([])  # desired track angle [deg]
            self.vs  = np.array([])  # desired vertical speed [m/s]
            self.spd = np.array([])  # desired speed [m/s]

    def create(self):
        super(Pilot, self).create()

        self.alt[-1] = self.traf.alt[-1]
        self.spd[-1] = tas2eas(self.traf.tas[-1], self.traf.alt[-1])
        self.hdg[-1] = self.traf.hdg[-1]
        self.trk[-1] = self.traf.trk[-1]

    def FMSOrAsas(self):
        #--------- Input to Autopilot settings to follow: destination or ASAS ----------
        # Convert the ASAS commanded speed from ground speed to TAS
        if self.traf.wind.winddim > 0:
            vwn, vwe     = self.traf.wind.getdata(self.traf.lat, self.traf.lon, self.traf.alt)
            asastasnorth = self.asas.spd * np.cos(np.radians(self.asas.trk)) - vwn
            asastaseast  = self.asas.spd * np.sin(np.radians(self.asas.trk)) - vwe
            asastas      = np.sqrt(asastasnorth**2 + asastaseast**2)
        # no wind, then ground speed = TAS
        else:
            asastas = self.asas.spd

        # Determine desired states from ASAS or AP. Select asas if there is a conflict AND resolution is on.
        self.trk = np.where(self.asas.active, self.asas.trk, self.ap.trk)
        self.spd = np.where(self.asas.active, asastas, self.ap.tas)
        self.alt = np.where(self.asas.active, self.asas.alt, self.ap.alt)
        self.vs  = np.where(self.asas.active, self.asas.vs, self.ap.vs)
        
        # ASAS can give positive and negative VS, but the sign of VS is determined using delalt in Traf.ComputeAirSpeed 
        # Therefore, ensure that pilot.vs is always positive to prevent opposite signs of delalt and VS in Traf.ComputeAirSpeed 
        self.vs = np.abs(self.vs)

        # Compute the desired heading needed to compensate for the wind
        if self.traf.wind.winddim > 0:
            # Calculate wind correction
            vwn, vwe = self.traf.wind.getdata(self.traf.lat, self.traf.lon, self.traf.alt)
            Vw       = np.sqrt(vwn * vwn + vwe * vwe)
            winddir  = np.arctan2(vwe, vwn)
            drift    = np.radians(self.trk) - winddir  # [rad]
            steer    = np.arcsin(np.minimum(1.0, np.maximum(-1.0,
                                                         Vw * np.sin(drift) / np.maximum(0.001, self.traf.tas))))
            # desired heading
            self.hdg = (self.trk + np.degrees(steer)) % 360.
        else:
            self.hdg = self.trk % 360.

    def FlightEnvelope(self):
        # check for the flight envelope
        self.traf.delalt = self.traf.apalt - self.traf.alt  # [m]
        self.traf.perf.limits()

        # Update desired sates with values within the flight envelope
        # To do: add const Mach const CAS mode
        self.spd = np.where(self.traf.limspd_flag, vcas2tas(self.traf.limspd, self.traf.alt), self.spd)

        # Autopilot selected altitude [m]
        self.alt = np.where(self.traf.limalt > -900, self.traf.limalt, self.alt)

        # Autopilot selected vertical speed (V/S)
        self.vs = np.where(self.traf.limvs > -9000, self.traf.limvs, self.vs)

        # To be discussed: Following change in VNAV mode only?
        # below crossover altitude: CAS=const, above crossover altitude: MA = const
        # climb/descend above crossover: Ma = const, else CAS = const
        # ama is fixed when above crossover
        self.traf.ama = np.where(self.traf.abco * (self.traf.ama == 0.),
                                 vcas2mach(self.traf.aspd, self.traf.alt), self.traf.ama)

        # ama is deleted when below crossover
        self.traf.ama = np.where(self.traf.belco, 0.0, self.traf.ama)
