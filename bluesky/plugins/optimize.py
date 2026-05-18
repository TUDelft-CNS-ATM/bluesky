import openap
import numpy as np
import pandas as pd

# This plugin requires opentop and its dependencies. Install them with:
#   uv pip install opentop
import opentop

from bluesky import core, stack, traf


DEFAULT_WIND_ALTITUDES_FT = (0.0, 10000.0, 20000.0, 30000.0, 45000.0)


class _OptimizeTestWind:
    winddim = 1

    def getdata(self, lat, lon, alt):
        return np.full(len(lat), 8.0), np.full(len(lon), -3.0)


def init_plugin():
    """Plugin initialisation function."""
    optimization = Optimization()
    config = {
        "plugin_name": "OPTIMIZATION",
        "plugin_type": "sim",
    }
    return config


def bluesky_wind_grid(
    wind,
    lat0,
    lon0,
    lat1,
    lon1,
    *,
    lat_points=5,
    lon_points=5,
    altitudes_ft=DEFAULT_WIND_ALTITUDES_FT,
):
    """Sample a static BlueSky wind field into the grid schema expected by opentop."""
    lat_margin = max(1.0, abs(lat1 - lat0) * 0.1)
    lon_margin = max(1.0, abs(lon1 - lon0) * 0.1)
    latitudes = np.linspace(
        min(lat0, lat1) - lat_margin, max(lat0, lat1) + lat_margin, lat_points
    )
    longitudes = np.linspace(
        min(lon0, lon1) - lon_margin, max(lon0, lon1) + lon_margin, lon_points
    )
    altitudes_m = np.asarray(altitudes_ft, dtype=float) * 0.3048

    lat_grid, lon_grid, alt_grid = np.meshgrid(
        latitudes, longitudes, altitudes_m, indexing="ij"
    )
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()
    flat_alt = alt_grid.ravel()
    vnorth, veast = wind.getdata(flat_lat, flat_lon, flat_alt)

    return pd.DataFrame(
        {
            "ts": 0.0,
            "latitude": flat_lat,
            "longitude": flat_lon,
            "h": flat_alt,
            "u": veast,
            "v": vnorth,
        },
        columns=["ts", "latitude", "longitude", "h", "u", "v"],
    )


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
        optimizer = opentop.CompleteFlight(
            actype, (lat0, lon0), (lat1, lon1), m0=m0, use_synonym=True
        )
        if traf.wind.winddim > 0:
            optimizer.enable_wind(bluesky_wind_grid(traf.wind, lat0, lon0, lat1, lon1))
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

    @stack.command(name="OPTTEST")
    def test(self):
        """Test the opentop wind interface used by the optimization plugin."""
        wind = bluesky_wind_grid(_OptimizeTestWind(), 52.0, 4.0, 51.5, 0.0)
        expected_columns = ["ts", "latitude", "longitude", "h", "u", "v"]
        if list(wind.columns) != expected_columns:
            return False, f"Unexpected wind columns: {list(wind.columns)}"
        if wind.empty:
            return False, "Wind grid is empty."
        if wind["u"].iloc[0] != -3.0 or wind["v"].iloc[0] != 8.0:
            return False, "BlueSky wind components were not mapped to opentop u/v."

        optimizer = opentop.CompleteFlight(
            "A320", (52.0, 4.0), (51.5, 0.0), m0=0.85, use_synonym=True
        )
        optimizer.enable_wind(wind)
        wu = optimizer.wind.calc_u(0.0, 0.0, 10000.0, 0.0)
        wv = optimizer.wind.calc_v(0.0, 0.0, 10000.0, 0.0)
        if not np.isclose(wu, -3.0) or not np.isclose(wv, 8.0):
            return False, f"Unexpected opentop wind values: u={wu}, v={wv}"

        return True, "OPTTEST passed: opentop wind interface is working."

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
