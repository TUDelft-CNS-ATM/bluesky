# PCALL — Import commands from another scenario file

```text
PCALL filename [REL/ABS] [, args...]
```

Merges another scenario file's commands into the one currently running
(alias: `CALL`), instead of resetting and replacing it the way [`IC`](ic.md)
does.

| Argument | Type | Required | Description |
|----------|------|----------|--------------|
| `filename` | text | yes | The scenario file to import |
| `REL`/`ABS` | text | no | Whether timestamps in the imported file are relative to the moment of the `PCALL` (the default), or absolute |
| `args...` | text | no | Positional arguments substituted for `%0`, `%1`, ... in the imported file |

## Examples

```text
PCALL approach-procedure.scn
PCALL emergency-descent.scn,KL204
```

In the second example, `%0` anywhere in `emergency-descent.scn` is replaced
with `KL204`.

## Notes

- `PCALL` supports a convenience form for per-aircraft procedure files:
  `PCALL KL204 myproc` (aircraft id first) works the same as `PCALL myproc,
  KL204`.
- Because timestamps default to being interpreted relative to the moment
  `PCALL` runs, the same sub-scenario can be triggered at different points
  in a run (or from different scenarios entirely) and its internal timing
  stays correct.
- See [Scenario files](../../user-guide/scenario-files.md) for the full
  scenario file format, and [`SCHEDULE`](../commands/index.md)/[`DELAY`](../commands/index.md)
  for inserting single commands at a future time without a whole file.
