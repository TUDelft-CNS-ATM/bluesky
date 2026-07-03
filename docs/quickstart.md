# Quickstart

This page walks through your first few minutes with BlueSky: starting it up,
creating an aircraft, and running a scenario.

## Starting BlueSky

After [installing](installation.md), start BlueSky with:

```console
$ bluesky
```

This opens the Qt/OpenGL window: a radar view of the world (centered on
Amsterdam Schiphol, `EHAM`, by default) with a command console at the bottom.
See [A tour of the user interface](user-guide/ui-tour.md) for a full
walkthrough of the window.

## Creating your first aircraft

Click into the console at the bottom of the window and type:

```text
CRE KL204,B738,52.3,4.76,90,FL200,280
```

This creates an aircraft `KL204`, a Boeing 737-800, at latitude 52.3,
longitude 4.76, heading 90 degrees, at flight level 200, at a speed of 280
knots. It should appear on the radar and start flying east.

You don't have to type coordinates by hand: type `CRE KL204,B738` and then
click a point on the radar to fill in the latitude/longitude — see
[radar-click command completion](user-guide/ui-tour.md#radar-click-command-completion).

Give it a destination and a route:

```text
DEST KL204,EHAM
ADDWPT KL204,EEL,FL180,250
```

Start (or resume) the simulation with:

```text
OP
```

## Loading a scenario

Rather than typing commands by hand, BlueSky is usually driven by **scenario
files** (`.scn`): text files with a timestamped command on every line. Try
one of the examples shipped with the repository:

```text
IC 0-demo-scenario.scn
```

`IC` (initial condition) resets the simulation and loads the given scenario.
Scenario file paths are resolved relative to the `scenario/` directory. See
[Scenario files](user-guide/scenario-files.md) for the full file format.

## Basic simulation control

A handful of commands you'll use constantly:

| Command | Effect |
|---------|--------|
| `OP` | Start/resume the simulation |
| `HOLD` | Pause the simulation |
| `RESET` (or `IC IC`) | Reset the simulation / reload the last scenario |
| `FF [time]` | Fast-forward, optionally to a given time |
| `DTMULT n` | Run at `n` times real-time speed |
| `QUIT` | Close BlueSky |

## Where to go next

- [A tour of the user interface](user-guide/ui-tour.md) — radar view, console, keyboard and mouse
- [Stack command syntax](user-guide/command-syntax.md) — how command arguments work
- [Stack command reference](reference/commands/index.md) — every command BlueSky supports
- [Writing your first plugin](plugins/first-plugin.md) — extend BlueSky with your own logic
