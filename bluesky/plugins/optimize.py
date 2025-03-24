import openap
from openap import top

from bluesky import core, stack, traf


def init_plugin():
    """Plugin initialisation function."""
    optimization = Optimization()
    config = {
        "plugin_name": "OPTIMIZATION",
        "plugin_type": "sim",
    }
    return config


class Optimization(core.Entity):
    def __init__(self):
        super().__init__()

    @stack.command
    def optimalflight(
        self,
        acid: "acid",
        actype: str,
        lat0: float,
        lon0: float,
        lat1: float,
        lon1: float,
        m0: float = 0.85,
    ):
        """Optimize flight plan for aircraft 'acid' and add waypoints using CRE/ADDWPT.

        Parameters:
            acid: Aircraft ID
            actype: Aircraft type.
            origin_lat: Latitude for origin.
            origin_lon: Longitude for origin.
            dest_lat: Latitude for destination.
            dest_lon: Longitude for destination.
        """
        optimizer = top.CompleteFlight(
            actype, lat0, lon0, lat1, lon1, m0=m0, use_synonyms=True
        )
        flight = optimizer.trajectory()

        first = flight.iloc[0]
        stack.stack(
            f"CRE {traf.id[acid]},{actype},{first.latitude},{first.longitude},"
            f"{first.heading} {first.altitude} {first.tas}"
        )

        for _, point in flight.iloc[1:].iterrows():
            lat = point["latitude"]
            lon = point["longitude"]
            alt = point["altitude"]
            spd = point["tas"]
            stack.stack(f"ADDWPT {traf.id[acid]} {lat} {lon} {alt} {spd}")

        return True, "Optimized flight plan generated."

    @stack.command
    def optimalflightod(self, acid: "acid", actype: str, origin: str, destination: str):
        """
        Optimize flight plan for aircraft 'acid' using ICAO airpoint codes for origin and destination.
        Retrieves lat/lon from openap.nav.airport.
        """
        # Retrieve airport data
        origin_airport = openap.nav.airport(origin)
        dest_airport = openap.nav.airport(destination)
        lat0, lon0 = origin_airport["lat"], origin_airport["lon"]
        lat1, lon1 = dest_airport["lat"], dest_airport["lon"]

        return self.optimalflight(acid, actype, lat0, lon0, lat1, lon1)
