# ASAS — Toggle conflict detection

```text
ASAS ON/OFF
```

Turns the Airborne Separation Assurance System's conflict *detection* on or
off (alias: `CDMETHOD`, which can also select the detection method — see
below). Called without arguments, it reports the current state.

| Argument | Type | Description |
|----------|------|--------------|
| `ON`/`OFF` | bool | Toggle conflict detection |

## Examples

```text
ASAS ON
ASAS OFF
```

## Notes

- Turning `ASAS` off disables conflict *detection* entirely, which means
  [`RESO`](reso.md) (resolution) has nothing to act on either — use
  [`RESOOFF`](../commands/index.md) instead if you want detection to keep
  running (e.g. for logging/analysis) without aircraft actually maneuvering
  to resolve conflicts.
- Related tuning commands: `ZONER`/`ZONEDH` (protected zone size),
  `DTLOOK` (lookahead time), `DTNOLOOK` (minimum time before a conflict is
  flagged).
- See [Conflict detection and
  resolution](../../concepts/conflict-detection-resolution.md) for the full
  picture, including how detection and resolution are independently
  replaceable.
