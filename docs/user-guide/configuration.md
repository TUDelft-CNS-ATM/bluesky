# Configuration

BlueSky's configuration lives in `settings.cfg`, resolved and loaded by
`bluesky/settings.py` and `bluesky/pathfinder.py`.

## The working directory

On startup, BlueSky determines a **working directory** that holds your
scenarios, plugins, output logs, and cache:

- **Running from a source checkout** (e.g. `python BlueSky.py` from the
  cloned repository) — the working directory is the repository root itself:
  `scenario/`, `plugins/`, `output/`.
- **Running as an installed pip package** — the working directory defaults
  to `~/bluesky`. On first run, BlueSky creates this directory along with
  `scenario/`, `plugins/`, `output/` and `cache/` subdirectories, and copies
  in a default `settings.cfg`.
- **Explicit override** — pass `--workdir DIR` on the command line, or
  `workdir=` to `bs.init()` when using BlueSky as a library.

## `settings.cfg`

`settings.cfg` is not a static key/value format — it is **executed as
Python source** (`settings.init()` does `exec()` on its contents into the
settings module's namespace). This means values can be any Python
expression, e.g. a list:

```python
enabled_plugins = ['area', 'datafeed']
performance_model = 'openap'
simdt = 0.05
```

If the file doesn't exist yet, it's copied from `bluesky/resources/default.cfg`.

### Some commonly used settings

| Setting | Default | Meaning |
|---------|---------|---------|
| `simdt` | `0.05` | Base simulation time step, in seconds |
| `performance_model` | `'openap'` | Aircraft performance model: `'openap'`, `'bada'`, or `'legacy'` |
| `asas_dt` | `1.0` | Conflict detection update interval |
| `plugin_path` | `'plugins'` | Directory (relative to the working directory) BlueSky scans for plugins |
| `enabled_plugins` | `['area', 'datafeed']` | Plugins loaded automatically on startup |
| `scenario_path` | `'scenario'` | Base directory for resolving scenario filenames |
| `start_location` | `'EHAM'` | Where the radar view starts panned/zoomed to |
| `recv_port` / `send_port` | `11000` / `11001` | ZeroMQ ports used for node/client networking |
| `max_nnodes` | *(CPU count)* | Maximum number of simulation nodes a server will spawn |

See the [settings reference](../reference/settings-reference.md) for a
fuller list, and [Using plugins](../plugins/using-plugins.md) for how
`enabled_plugins` and `plugin_path` interact with the other two ways of
activating a plugin.

## Modules register their own defaults

Individual modules and plugins declare their own settings (and defaults)
with `settings.set_variable_defaults(...)`, only taking effect if the
setting isn't already defined in your `settings.cfg`. This is why enabling a
plugin can introduce new settings you didn't have to declare yourself — the
plugin adds sensible defaults for them.

## Editing settings

- Edit `settings.cfg` directly in a text editor (it's plain, if
  Python-flavoured, text), or
- Use the **settings window** in the Qt GUI to change values interactively
  at runtime.

Use `--configfile FILE` on the command line to load an alternative
configuration file instead of the default `settings.cfg` in your working
directory — handy for switching between different setups (e.g. one config
per experiment) without editing files by hand.
