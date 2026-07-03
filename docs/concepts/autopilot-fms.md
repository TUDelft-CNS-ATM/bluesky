# Autopilot, FMS and routes

Flight guidance is split between the **autopilot** (selected
altitude/speed/heading, LNAV/VNAV logic) and each aircraft's **route** (its
flight plan / FMS waypoint list).

## Autopilot

`bs.traf.ap` is a replaceable `Autopilot` entity
(`bluesky/traffic/autopilot.py`) that computes each aircraft's selected
targets every simulation step. It registers most of the "manual guidance"
stack commands:

| Command | Effect |
|---------|--------|
| `ALT acid, alt, [vspd]` | Select an altitude (and optionally a vertical speed) |
| `VS acid, vspd` | Select a vertical speed directly |
| `HDG` (alias `HEADING`, `TURN`) | Select a heading |
| `SPD` (alias `SPEED`) | Select a speed (CAS or Mach) |
| `DEST acid, apt/latlon` | Set the destination |
| `ORIG acid, apt/latlon` | Set the origin |
| `VNAV acid, ON/OFF` | Toggle vertical navigation (follow route altitude/speed constraints) |
| `LNAV acid, ON/OFF` | Toggle lateral navigation (follow the route laterally) |
| `SWTOC` / `SWTOD` | Toggle top-of-climb / top-of-descent calculations |

When `LNAV`/`VNAV` are on, the autopilot's selected targets are derived from
the aircraft's active route and its waypoint constraints rather than from
manually selected values; issuing `HDG`/`ALT`/`SPD` directly typically
disengages the corresponding navigation mode, the same way it would on a
real aircraft's mode control panel.

## Active waypoint

`bs.traf.actwp` (`ActiveWaypoint`, also replaceable) tracks each aircraft's
*current* target waypoint and turn geometry — when a waypoint counts as
`reached()`, and how to calculate the turn onto the next leg
(fly-by vs. fly-over waypoints affect this).

## Routes

Each aircraft has its own `Route` object (`bluesky/traffic/route.py`) — its
ordered list of waypoints, with optional altitude/speed constraints at each
one. Route-editing commands:

| Command | Effect |
|---------|--------|
| `ADDWPT acid, (wpname/lat,lon), [alt,spd,afterwp]` | Add a waypoint to the route |
| `AT acid, wpname, [DEL], SPD/ALT, [value]` | Edit, show, or delete a constraint at a waypoint |
| `DIRECT acid, wpname` (aliases `DIRECTTO`, `DIRTO`, `DCT`) | Go direct to a waypoint, skipping any waypoints before it |
| `RTA` | Set a required time of arrival at a waypoint |
| `DELRTE acid` (alias `DELROUTE`) | Delete an aircraft's entire route |
| `DELWPT acid, wpname` (alias `DELWP`) | Remove a single waypoint |

Waypoints can reference the navigation database (airports, navaids,
airways) or be arbitrary lat/lon coordinates; see the `wpt`/`wpinroute`
[argument types](../user-guide/command-syntax.md).

## Worked example

```text
CRE KL204,B738,52.3,4.76,90,FL200,280
ORIG KL204,EHAM
DEST KL204,EDDF
ADDWPT KL204,EEL,FL180,250
ADDWPT KL204,DEGES
VNAV KL204,ON
LNAV KL204,ON
```

See the [`ADDWPT`](../reference/commands/addwpt.md) and
[`DEST`](../reference/commands/dest.md) command deep dives for more detail.
