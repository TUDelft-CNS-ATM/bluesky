# SPD — Select a speed

```text
SPD acid,spd
```

Selects a target speed for the autopilot, in knots calibrated airspeed
(CAS) or Mach.

| Argument | Type | Description |
|----------|------|--------------|
| `acid` | text | Aircraft id |
| `spd` | float | Target speed. A value below about 1 (e.g. `0.78`) is interpreted as Mach; larger values are knots CAS |

## Examples

```text
SPD KL204,250
SPD KL204,0.78
```

## Notes

- As with [`ALT`](alt.md), selecting a speed directly typically disengages
  `VNAV`'s own speed-constraint following for that aircraft — see
  [Autopilot, FMS and routes](../../concepts/autopilot-fms.md).
- The aircraft's [performance model](../../concepts/performance-models.md)
  still enforces a realistic speed envelope (`vmin`/`vmax`); an
  out-of-envelope `SPD` request is clamped.
