# HDG — Select a heading

```text
HDG acid,hdg [,deg/True]
```

Selects a target heading for the autopilot (aliases: `HEADING`, `TURN`).

| Argument | Type | Required | Description |
|----------|------|----------|--------------|
| `acid` | text | yes | Aircraft id |
| `hdg` | float | yes | Target heading, degrees |
| type | `deg`/`True` | no | Heading reference (true vs. magnetic), where supported |

## Examples

```text
HDG KL204,270
```

## Notes

- You can set the heading by clicking the radar view instead of typing a
  number: type `HDG KL204` and click in the direction you want the aircraft
  to turn toward — see [radar-click command
  completion](../../user-guide/ui-tour.md#radar-click-command-completion).
- Selecting `HDG` directly typically disengages `LNAV` for that aircraft —
  see [Autopilot, FMS and routes](../../concepts/autopilot-fms.md). Use
  [`DIRECT`](../commands/index.md) instead if you want to resume the FMS
  route at a specific waypoint rather than flying a fixed heading.
