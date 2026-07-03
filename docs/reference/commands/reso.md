# RESO — Set the conflict resolution method

```text
RESO [method]
```

Selects which conflict-resolution algorithm is active.

| Argument | Type | Required | Description |
|----------|------|----------|--------------|
| `method` | `OFF`/`MVP`/`EBY`/... | yes | Conflict resolution method |

Built-in and example methods:

| Method | Description |
|--------|--------------|
| `OFF` | Do nothing (detection may still run, but no maneuvers are applied) |
| `MVP` | Modified Voltage Potential method |
| `EBY` | Method by Martin S. Eby, shipped as a plugin (`bluesky/plugins/asas/eby.py`) |

Additional methods can be added by plugins — see [Replaceable
entities](../../plugins/replaceable-entities.md), which walks through how
`EBY` itself is implemented as a drop-in `ConflictResolution` replacement.

## Example

```text
RESO EBY
```

## Related

- `RESO` is really shorthand for selecting a `ConflictResolution`
  implementation; the general mechanism is `IMPL CONFLICTRESOLUTION <name>`
  — see [Replaceable entities](../../plugins/replaceable-entities.md).
- Tuning: `RFACH`/`RFACV` (resolution margin factors), `RSZONER`/`RSZONEDH`
  (resolution zone size, independent of the detection protected zone),
  `PRIORULES` (right-of-way rules), `NORESO acid` (exempt one aircraft).
- See [Conflict detection and
  resolution](../../concepts/conflict-detection-resolution.md) for how
  detection and resolution relate.
