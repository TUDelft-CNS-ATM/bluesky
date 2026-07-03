# Command line options

`bluesky` (or `python BlueSky.py` from a source checkout) accepts the
following command line options.

## Mode selection

These flags are mutually exclusive. With no flag, BlueSky starts a
simulation **server with the Qt GUI** (the normal desktop experience).

| Flag | Effect |
|------|--------|
| *(none)* | Start simulation server + Qt GUI in one process |
| `--headless` | Start simulation server only, without a GUI, and make it discoverable |
| `--client [hostname]` | Start a Qt GUI client connecting to a running server. Without a hostname, a discovery dialog lets you pick a server |
| `--console [hostname]` | Start a text-based console client connecting to a running server |
| `--sim` | Start a single simulation node (networked, for a server to connect to) |
| `--detached` | Start a single simulation node with **no networking**, for embedding BlueSky in another Python script |

See [Running modes](../user-guide/running-modes.md) for when to use each of these.

## Other options

| Flag | Effect |
|------|--------|
| `scenfile` (positional) or `--scenfile FILE` | Load a scenario file on startup |
| `--configfile FILE` | Load an alternative configuration file instead of `settings.cfg` |
| `--discoverable` | Make the simulation server discoverable via UDP (on by default with `--headless`) |
| `--workdir DIR` | Use a custom BlueSky working directory instead of the current directory or `~/bluesky` |
| `--groupid ID` | Explicitly set (part of) the network connection identifier, instead of a random one |

## Examples

```console
$ bluesky
$ bluesky --headless
$ bluesky --client
$ bluesky scenario/0-demo-scenario.scn
$ bluesky --detached --workdir /tmp/bluesky-run
```
