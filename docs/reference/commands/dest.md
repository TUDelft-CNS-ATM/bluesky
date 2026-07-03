# DEST — Set an aircraft's destination

```text
DEST acid, latlon/airport
```

Sets an aircraft's destination, either as an airport (4-letter ICAO code)
or a raw lat/lon position.

| Argument | Type | Description |
|----------|------|--------------|
| `acid` | text | Aircraft id |
| `airport` | text | 4-letter ICAO airport code |
| `lat`, `lon` | float | Destination position, if not using an airport code |

## Examples

```text
DEST KL204,EHAM
DEST KL204,52.3086,4.7639
```

## Related commands

- [`ORIG`](../commands/index.md) — set the origin airport, the counterpart to `DEST`
- [`ADDWPT`](addwpt.md) — build out the full route, not just the endpoint
- Setting `DEST` alone (without a route) is enough for the autopilot to fly
  direct to it once [`LNAV`/`VNAV`](vnav.md) are enabled.
