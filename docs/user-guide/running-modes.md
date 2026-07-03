# Running modes

BlueSky can run as a single desktop application, or split across separate
server, simulation-node, and client processes. The mode is selected with
[command line flags](../reference/cli.md) (`bluesky/cmdargs.py`) and is
validated/wired up by `bs.init()` (`bluesky/__init__.py`).

## Process topology

- **Server** (`bluesky/network/server.py`) — manages one or more simulation
  node subprocesses, routes messages between nodes and clients over
  ZeroMQ/msgpack, and can be made discoverable on the local network via UDP.
- **Node** (`bluesky/network/node.py`) — a single running simulation
  (`bs.sim`, `bs.traf`, ...). A server can spawn multiple nodes, e.g. for
  parallel batch runs (`ADDNODES`).
- **Client** — a GUI (Qt or console) that connects to a server and displays
  the state of one of its nodes.

## Modes

| Flag | Process(es) started | Typical use |
|------|----------------------|-------------|
| *(none)* | Server + node + Qt GUI, all in one process | Normal interactive desktop use |
| `--headless` | Server + node, no GUI, discoverable | Running on a remote machine / in a container |
| `--client [host]` | Qt GUI client only | Connecting a GUI to an already-running (possibly remote) server |
| `--console [host]` | Text console client only | Lightweight/remote interaction without a GUI |
| `--sim` | A single simulation node, networked | Adding a node to an existing server, or scripted multi-node setups |
| `--detached` | A single simulation node, **no networking at all** | Embedding BlueSky inside another Python program (see [Using BlueSky as a library](../api/index.md)) |

Without a mode flag, `hostname` connection details, `--discoverable` and
`--groupid` don't apply; they matter once you start splitting the server,
nodes and clients across machines or processes.

## Multi-node batch runs

The `BATCH` command runs a list of scenarios unattended, distributing them
across as many simulation nodes as the server has started (up to
`max_nnodes`, which defaults to the machine's CPU count). Use `ADDNODES n`
to add simulation nodes to a running server, e.g. to parallelize a large
batch of scenarios across cores.

## Choosing a mode

- Interactively exploring or building scenarios on your own machine: default
  mode (no flags).
- Running large batches of experiments without a GUI: `--headless`, driving
  it with `BATCH` or by connecting a client later to inspect it.
- Embedding BlueSky's simulation core in your own Python script or notebook
  (e.g. to script traffic generation, or wrap it as a gym-style RL
  environment): `--detached`, or equivalently `bs.init(mode='sim',
  detached=True)` from Python — see [Using BlueSky as a
  library](../api/index.md).
