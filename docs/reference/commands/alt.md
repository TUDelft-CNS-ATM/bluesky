# ALT — Select an altitude

```text
ALT acid, alt, [vspd]
```

Selects a target altitude for the autopilot, and optionally a specific
vertical speed to climb or descend at.

| Argument | Type | Required | Description |
|----------|------|----------|--------------|
| `acid` | text | yes | Aircraft id |
| `alt` | float | yes | Selected altitude, feet (or `FLnnn`) |
| `vspd` | float | no | Selected climb/descend rate, ft/min |

## Examples

```text
ALT KL204,FL200
ALT KL204,5000,-1500
```

## Notes

- Without `vspd`, the aircraft climbs or descends at a rate determined by
  its [performance model](../../concepts/performance-models.md).
- Issuing `ALT` directly selects a manual altitude target — if the aircraft
  has `VNAV` enabled, this typically disengages vertical navigation the
  same way selecting an altitude on a real aircraft's mode control panel
  would. See [Autopilot, FMS and routes](../../concepts/autopilot-fms.md).
- See [`VS`](../commands/index.md) to select a vertical speed directly
  without changing the target altitude.
