# AREA — Define an experiment area

```text
AREA Shapename/OFF
AREA lat1,lon1,lat2,lon2,[top,bottom]
```

Defines the "area of interest" for the simulation. When an aircraft exits
this area, it is deleted — commonly used to bound an experiment region so
aircraft that have flown out of the area of study don't keep consuming
simulation resources.

| Form | Description |
|------|--------------|
| `AREA shapename` | Use an already-defined shape (see `BOX`, `CIRCLE`, `POLY`, `POLYALT`) as the experiment area |
| `AREA lat1,lon1,lat2,lon2,[top,bottom]` | Define a rectangular area directly, optionally 3D with altitude bounds |
| `AREA OFF` | Disable the area filter (aircraft are no longer deleted on exit) |

## Examples

```text
BOX EXPBOX,52.0,3.5,53.0,5.0
AREA EXPBOX
```

```text
AREA 52.0,3.5,53.0,5.0,FL300,0
```

## The `AREA` plugin and logging

The shipped `area` plugin (`bluesky/plugins/area.py`, enabled by default —
see `enabled_plugins` in [Configuration](../../user-guide/configuration.md))
builds on the `AREA` command to log flight statistics — distance flown, time
in the area, fuel used — for every aircraft when it exits, using the
[data logger](../../user-guide/logging-and-output.md). This makes `AREA`
useful both as a spatial filter and as the trigger point for per-flight
metrics collection in a typical batch experiment.

## Related

- `BOX` / `CIRCLE` / `POLY` / `POLYALT` — define reusable named shapes
- [Logging and output](../../user-guide/logging-and-output.md)
