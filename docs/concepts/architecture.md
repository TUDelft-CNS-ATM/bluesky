# Architecture overview

BlueSky is built around a small set of global singleton objects, a
networked server/node/client topology, and a data-parallel traffic model.
This page is the map; later pages in this section go into each piece.

## The `bs.*` globals

`import bluesky as bs` gives access to a handful of module-level singletons,
populated by `bs.init(mode=...)`:

| Global | What it is |
|--------|------------|
| `bs.sim` | The `Simulation` object — controls the simulation loop and time |
| `bs.traf` | The `Traffic` object — all aircraft state (see [The traffic model](traffic-model.md)) |
| `bs.navdb` | The navigation database (navaids, airports, airways) |
| `bs.scr` | The screen/`ScreenIO` interface, feeding aircraft data to clients |
| `bs.net` | The networking node/client (see [Networking](networking.md)) |
| `bs.server` | The `Server` object, only set in server mode |
| `bs.ref` | A scratch reference position used while parsing command arguments |
| `bs.settings`, `bs.stack`, `bs.tools` | Modules, not singletons, but commonly accessed the same way |

Because these are populated once at `bs.init()` time and referenced by name
everywhere else in the codebase, code (including plugins) can simply
`import bluesky as bs` and use `bs.traf`, `bs.sim`, etc. without needing them
passed around explicitly.

## Simulation states

The simulation is always in one of four states (`bs.INIT`, `bs.HOLD`,
`bs.OP`, `bs.END` — `INIT, HOLD, OP, END = range(4)`):

- **INIT** — before the first scenario has been loaded/started
- **HOLD** — paused (via the `HOLD` command)
- **OP** — running
- **END** — shutting down

See [The simulation loop and time](simulation-loop.md) for how these states
drive what runs on every tick.

## Server, node and client processes

For networked operation (all modes except `--detached`), BlueSky splits into
up to three kinds of process, connected over ZeroMQ with msgpack-encoded
messages:

- **Server** — routes messages between clients and simulation nodes, spawns
  and manages node subprocesses, and (optionally) advertises itself on the
  network for discovery.
- **Simulation node** — runs the actual simulation (traffic, autopilot,
  stack processing). A server can run several nodes at once, e.g. to
  distribute a batch of scenarios across CPU cores.
- **Client** — a GUI (Qt or console) that connects to a server and displays
  one node's state.

In the default (no-flag) startup mode, server, one node, and the Qt client
all run in a single process for simplicity — but the same message-passing
architecture is used internally either way. See [Running
modes](../user-guide/running-modes.md) for the practical implications, and
[Networking](networking.md) for the wire-level details.

## Replaceable implementations

Several core subsystems — the autopilot, conflict resolution, the
performance model — are implemented as **replaceable singletons**
(`bluesky.core.Entity` with `replaceable=True`). Callers always refer to
them the same way (e.g. `bs.traf.cr` for conflict resolution), while the
concrete implementation behind that reference can be swapped at runtime with
the `IMPL`/`IMPLEMENTATION` command, or by a plugin registering an
alternative implementation. This is how, for example, BlueSky ships several
interchangeable conflict-resolution algorithms and performance models
without their callers needing to know which one is active.
