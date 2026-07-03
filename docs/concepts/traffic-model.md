# The traffic model

`bluesky.traffic.traffic.Traffic` (`bs.traf`) holds the state of every
aircraft in the simulation, and drives their behaviour on every simulation
step.

## Aircraft state as parallel arrays

`Traffic` is the root of BlueSky's `TrafficArrays` system
(`bluesky/core/trafficarrays.py`): all aircraft state — id, type, lat, lon,
altitude, heading, track, speeds (TAS/CAS/GS/Mach), vertical speed,
selected autopilot targets, LNAV/VNAV switches, and so on — is stored as
*parallel NumPy arrays*, one element per aircraft, rather than one object
per aircraft. Creating or deleting aircraft automatically resizes every
registered array in sync, including arrays registered by plugins (see
[Traffic arrays](../plugins/traffic-arrays.md)).

This vectorized-by-construction design is why `Traffic.update()` can update
every aircraft's kinematics, autopilot logic, and conflict detection in a
handful of NumPy operations per tick, rather than looping over aircraft
objects in Python.

## Sub-entities

`Traffic` owns a number of sub-entities, each responsible for one part of
aircraft behaviour:

| Attribute | Class | Responsibility |
|-----------|-------|-----------------|
| `bs.traf.ap` | `Autopilot` (replaceable) | LNAV/VNAV logic, selected altitude/speed/heading — see [Autopilot, FMS and routes](autopilot-fms.md) |
| `bs.traf.actwp` | `ActiveWaypoint` (replaceable) | Active-waypoint state and turn geometry |
| *(per aircraft)* | `Route` | Each aircraft's waypoint list |
| `bs.traf.perf` | performance model | Aircraft performance — see [Aircraft performance models](performance-models.md) |
| `bs.traf.cd` / `bs.traf.cr` | `ConflictDetection` / `ConflictResolution` (replaceable) | See [Conflict detection and resolution](conflict-detection-resolution.md) |
| `bs.traf.aporasas` | — | Arbitrates between autopilot and ASAS commands |
| `bs.traf.adsb` | `ADSB` model | Simulated ADS-B surveillance |
| `bs.traf.wind` | `WindSim` | See [Wind, weather and ADS-B](wind-weather-adsb.md) |
| `bs.traf.turbulence` | — | Turbulence model |
| `bs.traf.trails` | — | Flown-track trails |
| `bs.traf.cond` | `Condition` | `AT*` conditional commands (`ATALT`, `ATDIST`, `ATSPD`) |
| `bs.traf.groups` | `TrafficGroups` | Named aircraft groups (`GROUP`/`UNGROUP`) |
| `bs.traf.metric` | — | Traffic complexity/density metrics |

## The per-tick update pipeline

`Traffic.update()` runs, in order, once per simulation step (when the
simulation is in the `OP` state):

1. Update atmosphere (pressure, density, temperature at each aircraft's
   position/altitude)
2. Update the simulated ADS-B model
3. `ap.update()` — autopilot/FMS logic
4. Conflict detection and resolution (gated by their own timer, since these
   don't need to run every base time step)
5. `aporasas.update()` — reconcile autopilot vs. ASAS-selected targets
6. `perf.limits()` — apply aircraft performance envelope limits
7. Kinematics — update airspeed, ground speed, and position
8. Turbulence
9. Conditional command checks (`AT*`)
10. Trails

## Key methods

- `cre(...)` / `mcre(...)` — create one or many aircraft (backing the `CRE`
  and `MCRE` commands)
- `delete(idx)` — remove an aircraft
- `id2idx(acid)` — resolve a callsign to its array index (used by the
  `acid` [argument type](../user-guide/command-syntax.md))
- `move(...)` — teleport an aircraft to a new state (backing `MOVE`)
