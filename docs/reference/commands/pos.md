# POS — Get info on an object

```text
POS acid/waypoint/airport
```

Shows detailed information about an aircraft, airport, or waypoint. The
information shown depends on the object's type:

- **Aircraft** — position, heading, altitude, CAS/TAS/ground speed/Mach,
  LNAV/VNAV status, origin and destination.
- **Airport** — size, position, elevation, country, number of runways.
- **Waypoint** — type, position, description (including navaid
  frequencies), magnetic variation for VOR beacons, and a note if multiple
  waypoints share the same name.

## Examples

```text
POS KL204
POS EHAM
POS EEL
```

## Notes

- Typing an aircraft callsign as the *only* thing on the command line
  (i.e. not a recognized command) is treated as an implicit `POS` — see
  [Stack command syntax](../../user-guide/command-syntax.md).
- Double-clicking an aircraft on the radar view issues `POS` for it — see
  [the UI tour](../../user-guide/ui-tour.md).
