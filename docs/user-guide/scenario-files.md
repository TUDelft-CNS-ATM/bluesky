# Scenario files

BlueSky is normally driven by **scenario files** (`.scn`): plain text files
containing timestamped stack commands, played back by the simulation
(`bluesky/stack/simstack.py`). Hundreds of examples ship in the `scenario/`
directory.

## File syntax

Each line has the form:

```text
HH:MM:SS.hh>COMMAND args
```

- The timestamp before `>` is hours:minutes:seconds (seconds may have a
  fractional part) and is interpreted as **simulation time**, i.e. time
  since the scenario started.
- Everything after `>` is a normal stack command line — the same syntax you
  would type into the console.
- Lines are automatically sorted by timestamp when the file is loaded.
- Lines starting with `#` are comments and are skipped; an inline `#` also
  truncates the rest of a line.
- A trailing `\` continues a command onto the next line.

Example, from `scenario/0-demo-scenario.scn`:

```text
0:00:00.00>noise off
0:00:00.00>ASAS ON
0:00:00.00>pan eham
0:00:00.00>CRE MS841, B737, 51.42972, 4.15673, 118.4 FL320, 400
0:00:00.00>DEST OU106, EKKA
```

## Loading and running scenarios

| Command | Effect |
|---------|--------|
| `IC filename` (also `LOAD`/`OPEN`) | Reset the simulation and load a scenario file |
| `IC IC` | Reload the most recently run scenario |
| `IC` (no argument) | Open a file picker (Qt GUI only) |

Relative scenario filenames are resolved under the `scenario_path` setting
(default `scenario/`). Loading a scenario resets the simulation state first.

## Composing scenarios

- **`PCALL filename [, REL/ABS] [, args...]`** (alias `CALL`) — merges another
  scenario file into the one currently running, instead of replacing it.
  - `REL` (the default) interprets the sub-scenario's timestamps as relative
    to *now* (current simulation time); `ABS` keeps them as absolute times.
  - Positional arguments passed to `PCALL` are substituted into the
    sub-scenario wherever it references `%0`, `%1`, etc. — this lets you
    write parameterized, reusable scenario fragments.
- **`SCEN name`** (alias `SCENARIO`) — sets the current scenario's display name.
- **`SCHEDULE time, cmdline`** — inserts a command to run at an absolute
  future simulation time.
- **`DELAY time, cmdline`** — inserts a command to run a given number of
  seconds from now.
- **`BATCH filename`** — runs one or more scenarios as an unattended batch
  simulation, optionally distributed across [multiple simulation
  nodes](running-modes.md).

## Custom file formats

BlueSky can be taught to read scenario-like data from other file formats via
the `Importer` extension point (`bluesky/stack/importer.py`) — see [Custom
scenario importers](../plugins/custom-importers.md) if you need to load
flight plans or traffic from an existing dataset that isn't already `.scn`.
