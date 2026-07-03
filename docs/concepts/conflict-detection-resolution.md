# Conflict detection and resolution

BlueSky's Airborne Separation Assurance System (ASAS) is split into two
independent, replaceable stages: **detection** and **resolution**
(`bluesky/traffic/asas/`).

## Conflict detection

`bs.traf.cd` is a `ConflictDetection` entity (`detection.py`,
`replaceable=True`). Its job is purely to identify which aircraft pairs are
in conflict (predicted loss of separation within a lookahead time), not to
resolve them. Key commands:

| Command | Effect |
|---------|--------|
| `ASAS ON/OFF` (alias `CDMETHOD`) | Turn conflict detection on/off, or select the detection method |
| `ZONER` (alias `PZR`/`RPZ`/`PZRADIUS`) | Set the horizontal protected-zone radius |
| `ZONEDH` (alias `PZDH`/`DHPZ`/`PZHEIGHT`) | Set the vertical protected-zone half-height |
| `DTLOOK` | Set the conflict-detection lookahead time |
| `DTNOLOOK` | Set a minimum time before which conflicts aren't flagged (e.g. to ignore conflicts right after takeoff) |

The default detection method is **state-based** (`statebased.py`) —
extrapolating current aircraft states linearly to check for future
separation loss within the lookahead window.

## Conflict resolution

`bs.traf.cr` is a `ConflictResolution` entity (`resolution.py`,
`replaceable=True`). Given the conflicts found by detection, it computes
resolution maneuvers (heading/altitude/speed adjustments) to keep aircraft
separated. Key commands:

| Command | Effect |
|---------|--------|
| `RESO [method]` | Select the resolution method/algorithm |
| `RESOOFF` | Turn resolution off (detection can still run without acting on it) |
| `NORESO acid` | Exempt a specific aircraft from having resolution applied to it |
| `PRIORULES` | Set right-of-way / priority rules between conflicting aircraft |
| `RFACH` / `RFACV` (aliases `RESOFACH`/`RESOFACV`) | Horizontal/vertical resolution factor (how much margin to add beyond the minimum) |
| `RSZONER` / `RSZONEDH` (aliases `RESOZONER`/`RESOZONEDH`) | Resolution zone size, which can differ from the detection protected zone |

Two resolution algorithms ship as example implementations:

- **MVP** (`mvp.py`) — Modified Voltage Potential method
- **Eby** (`bluesky/plugins/asas/eby.py`) — a plugin-based alternative,
  demonstrating how a resolution algorithm can be added by *replacing*
  `ConflictResolution` rather than modifying it — see [Replaceable
  entities](../plugins/replaceable-entities.md).

## Writing your own

Because both `ConflictDetection` and `ConflictResolution` are replaceable
`Entity` subclasses, adding a new algorithm doesn't require modifying
existing code: subclass one of them in a plugin, override the relevant
methods (`detect()`/`resolve()`), and select it at runtime with
`IMPL`/`RESO`. See [Replaceable entities](../plugins/replaceable-entities.md)
for a worked walkthrough using `eby.py` as the example.
