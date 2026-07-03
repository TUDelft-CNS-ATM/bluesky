# IC — Load an initial condition (scenario)

```text
IC [IC/filename]
```

Resets the simulation and loads a new scenario file (aliases: `LOAD`,
`OPEN`).

| Argument | Description |
|----------|--------------|
| *(none)* | Opens a file picker (Qt GUI only) |
| `IC` (the literal word) | Reloads the current/most recently run scenario |
| `filename` | Loads the given `.scn` file |

## Examples

```text
IC 0-demo-scenario.scn
IC IC
```

Relative filenames are resolved under the `scenario_path` setting (default
`scenario/`).

## Notes

- `IC` fully resets simulation state before loading — aircraft, areas, and
  most settings from the previous run are cleared.
- See [Scenario files](../../user-guide/scenario-files.md) for the full
  `.scn` file format, and [`PCALL`](pcall.md) if you want to *merge* another
  scenario into the one currently running instead of replacing it.
- `RESET` resets the simulation without reloading a scenario file.
