# Logging and output

BlueSky can log simulation data to CSV-style files, and plot live variables
during a run.

## The data logger

`bluesky/tools/datalog.py` provides a generic CSV logger. Log files are
written under the `log_path` setting (default `output/`, inside your
[working directory](configuration.md)).

- **`CRELOG name, [dt], [header]`** — create a new logger. Passing `dt`
  makes it a *periodic* logger that writes a row every `dt` seconds of
  simulation time; without `dt`, the logger only writes when explicitly
  triggered.
- **`INSTLOG ON/OFF, [dt]`** — logger control; `INSTLOG LISTVARS` lists all
  variables currently available to log, and `INSTLOG SELECTVARS
  var1,var2,...` selects which ones to include.

Log filenames are automatically timestamped and include the current
scenario name, so successive runs don't overwrite each other.

Plugins commonly define their own loggers this way — see the
[`AREA` plugin](../reference/commands/area.md), which logs flight
statistics (distance flown, fuel used, etc.) whenever an aircraft leaves the
experiment area.

## Variable inspection

The **variable explorer** (`bluesky/core/varexplorer.py`, the `LSVAR`
command) lets you list and inspect the live Python variables of any loaded
module or plugin at runtime — useful for checking what data is available to
log or plot without reading source code.

## Plotting

The `PLOT`/`LEGEND` commands (`bluesky/tools/plotter.py`) draw live
matplotlib plots of simulation variables while the sim runs — handy for a
quick look at a metric during development, as an alternative to logging to
CSV and post-processing.

## Where output goes

- CSV logs: `output/` in your working directory (or wherever `log_path`
  points).
- Route dumps (`DUMPRTE`): also written under the working directory.
- Any file paths you pass to plugin-specific logging just follow the same
  [resource resolution](configuration.md) rules as everything else in
  BlueSky.
