# BATCH — Run scenarios as a batch simulation

```text
BATCH filename
```

Runs one or more scenarios unattended, as a batch — useful for large
experiment sweeps without manual interaction.

| Argument | Type | Description |
|----------|------|--------------|
| `filename` | text | Name of the batch file listing scenarios to run |

## How it fits together

When [running with multiple simulation nodes](../../user-guide/running-modes.md)
(e.g. a server that has spawned several nodes with `ADDNODES`), a batch is
automatically distributed across all available nodes, so independent
scenarios run in parallel rather than one after another.

## Related

- [`ADDNODES n`](../commands/index.md) — add simulation nodes to run a
  batch across more CPU cores
- [`BENCHMARK`](../commands/index.md) — run a single scenario as fast as
  possible and report timing
- [Running modes](../../user-guide/running-modes.md) — when `--headless`
  mode is the right fit for batch experiments
