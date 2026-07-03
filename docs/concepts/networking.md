# Networking

BlueSky's server, simulation nodes, and clients communicate over
**ZeroMQ**, with messages encoded using **msgpack** (`bluesky/network/`).
This is what makes the [server/node/client topology](architecture.md)
possible, whether all three run in one process or are spread across
machines.

## Nodes

`bluesky/network/node.py::Node` (`bs.net` inside a simulation process) is
the sim-side network endpoint. It uses a ZMQ `SUB` socket to receive and an
`XPUB` socket to send. Key operations:

- `connect()` — connect to a server
- `send(topic, data, to_group)` — publish a message
- `subscribe(topic, from_group, to_group, actonly)` / `unsubscribe(...)` —
  subscribe to a topic
- `receive()` / `update()` — pump incoming messages

Messages are addressed as `[to_group + topic + node_id, msgpack(data)]`;
incoming messages are dispatched to the topic's registered `Subscription`
signal. Default ports: `recv_port=11000`, `send_port=11001`.

## Server

`bluesky/network/server.py::Server` spawns and manages simulation node
subprocesses (up to `max_nnodes`, default the machine's CPU count), handles
`ADDNODES`, distributes batch scenarios across nodes (`split_scenarios`),
and acts as an XSUB/XPUB proxy between clients and nodes. It can optionally
run a UDP `Discovery` responder so clients on the same network can find it
without knowing its address in advance (`--discoverable`).

## Client

`bluesky/network/client.py::Client` is the endpoint used by GUIs to connect
to a server and receive simulation state to display.

## Publish/subscribe helpers

- `@subscriber(topic=...)` (`network/subscriber.py`) — decorate a function
  to auto-connect it to incoming messages on a topic.
- `@state_publisher(topic=...)` (`network/publisher.py`) — declare a
  function whose return value is periodically published under a topic
  (e.g. `STATECHANGE`, `STACKCMDS`, `SIMSETTINGS`).
- **Shared state** (`network/sharedstate.py`) — keeps a piece of state
  automatically synchronized between the sim side and connected clients,
  without hand-writing publish/subscribe boilerplate for every field.

## Detached mode

When BlueSky runs with `detached=True` (the `--detached` flag, or
`bs.init(mode='sim', detached=True)` when [using BlueSky as a
library](../api/index.md)), `bs.net` is replaced with a no-op stand-in
(`network/detached.py`) — no sockets are opened at all. This is what makes
detached mode suitable for embedding BlueSky in scripts, notebooks, or
automated documentation generation without any networking side effects.
