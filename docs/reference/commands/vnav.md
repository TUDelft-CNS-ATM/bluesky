# VNAV — Vertical navigation switch

```text
VNAV acid,switch
```

Turns vertical navigation (following the FMS route's altitude and speed
constraints) on or off for an aircraft.

| Argument | Type | Description |
|----------|------|--------------|
| `acid` | text | Aircraft id |
| `switch` | `ON`/`OFF` | Enable or disable VNAV |

## Example

```text
VNAV KL204,ON
```

## Notes

- The lateral counterpart is `LNAV` (same syntax: `LNAV acid,ON/OFF`),
  which makes the aircraft follow the route's lateral path rather than a
  fixed heading.
- With `VNAV ON`, altitude and speed at each waypoint follow whatever
  constraints were set via [`ADDWPT`](addwpt.md)/`AT`, rather than a
  manually selected [`ALT`](alt.md)/[`SPD`](spd.md) target.
- See [Autopilot, FMS and routes](../../concepts/autopilot-fms.md) for how
  VNAV, LNAV, and manual guidance commands interact, including top-of-climb
  and top-of-descent handling (`SWTOC`/`SWTOD`).
