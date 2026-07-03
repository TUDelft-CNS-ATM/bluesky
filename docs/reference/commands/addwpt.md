# ADDWPT — Add a waypoint to a route

```text
ADDWPT acid, (wpname/lat,lon), [alt,spd,afterwp]
```

Adds a waypoint to an aircraft's FMS route — either an existing waypoint
from the navigation database, or a custom lat/lon position.

| Argument | Type | Required | Description |
|----------|------|----------|--------------|
| `acid` | text | yes | Aircraft id |
| `wpname` / `lat,lon` | text / float | yes | Existing waypoint name, or a custom position |
| `FLYBY` / `FLYOVER` | text | no | Make this a fly-by or fly-over waypoint |
| `alt` | float | no | Altitude constraint at this waypoint |
| `spd` | float | no | Speed constraint at this waypoint |
| `afterwp` | text | no | Insert after this waypoint, instead of at the end of the route |

## Examples

```text
ADDWPT KL204,EEL,FL180,250
ADDWPT KL204,52.1,4.3
ADDWPT KL204,DEGES,,,EEL
```

The last example inserts `DEGES` right after `EEL` in the route, rather than
appending it at the end.

## Related commands

- [`AT`](../commands/index.md) — edit or clear a constraint on a waypoint
  already in the route
- [`DIRECT`](../commands/index.md) (aliases `DIRECTTO`/`DIRTO`/`DCT`) — skip
  ahead to a specific waypoint
- [`DELWPT`](../commands/index.md) / [`DELRTE`](../commands/index.md) —
  remove a waypoint, or the whole route
- [`VNAV`](vnav.md) / `LNAV` — enable the autopilot to actually follow the
  route's constraints and lateral path

See [Autopilot, FMS and routes](../../concepts/autopilot-fms.md) for how
routes fit into the bigger picture.
