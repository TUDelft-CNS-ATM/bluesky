# Wind, weather and ADS-B

A few smaller, independently replaceable subsystems round out the traffic
simulation: wind, turbulence, and simulated ADS-B surveillance.

## Wind

`bs.traf.wind` is a `WindSim` entity (`bluesky/traffic/windsim.py`,
replaceable). Wind affects ground speed and track versus airspeed and
heading throughout the kinematics update.

| Command | Effect |
|---------|--------|
| `WIND` | Define or clear a wind field (fixed value, or a grid) |
| `GETWIND lat,lon,[alt]` | Query the wind at a given position (and optionally altitude) |
| `DEL WIND` | Clear all defined wind |

Plugins can supply real weather data instead of hand-defined wind fields —
the repository includes plugins for pulling GFS and ECMWF wind/weather data
(`windgfs`, `windecmwf`) as examples of feeding external weather sources
into `WindSim`.

## Turbulence

`bs.traf.turbulence` (`bluesky/traffic/turbulence.py`, replaceable) adds
random perturbations to aircraft state, for more realistic (noisy) flight
paths. The `NOISE` command toggles it on/off.

## ADS-B model

`bs.traf.adsb` (`bluesky/traffic/adsbmodel.py`, replaceable) simulates
ADS-B surveillance — since BlueSky can also connect to *real* ADS-B/Mode S
data feeds (via the `datafeed` plugin, enabled by default), this model lets
you simulate the same kind of imperfect, delayed surveillance information
for purely synthetic traffic as well, rather than assuming every aircraft
has perfect, instantaneous knowledge of every other aircraft's state.

## Trails

`bs.traf.trails` (`bluesky/traffic/trails.py`) records each aircraft's
recently flown track for display as a fading trail on the radar view — a
display feature rather than a simulation model, but it hooks into the same
per-tick traffic update pipeline (see [The traffic
model](traffic-model.md)).
