# CRE — Create an aircraft

```text
CRE acid,type,lat,lon,hdg,alt,spd
```

Creates a single aircraft at the given position. See
[`MCRE`](../commands/index.md) to create multiple, randomly located
aircraft at once.

| Argument | Type | Description |
|----------|------|--------------|
| `acid` | text | Unique aircraft callsign |
| `type` | text | ICAO aircraft type designator (e.g. `B738`, `A320`) |
| `lat`, `lon` | float | Position |
| `hdg` | float | Heading, degrees |
| `alt` | float/text | Altitude, e.g. `FL200` or `5000` |
| `spd` | float | Speed, knots |

## Example

```text
CRE KL204,B738,52.3,4.76,90,FL200,280
```

Creates `KL204`, a 737-800, at 52.3°N 4.76°E, heading 090°, at FL200 and 280
knots.

## Tips

- You can fill in `lat,lon` (and `hdg`, via the click-drag direction) by
  clicking the radar view instead of typing coordinates — see [radar-click
  command completion](../../user-guide/ui-tour.md#radar-click-command-completion).
- Newly created aircraft have no route and no destination — follow up with
  [`ORIG`](index.md)/[`DEST`](dest.md) and [`ADDWPT`](addwpt.md) if you want
  it to fly a route, or [`HDG`](hdg.md)/[`ALT`](alt.md)/[`SPD`](spd.md) to
  fly manually selected targets.
- The aircraft type must exist in the active [performance
  model](../../concepts/performance-models.md); an unrecognized type will
  fail to create or fall back to a default.
