# Settings reference

The full contents of `bluesky/resources/default.cfg`, which is copied to
your working directory's `settings.cfg` the first time BlueSky runs — see
[Configuration](../user-guide/configuration.md) for how settings are loaded
and overridden.

## Network

| Setting | Default | Description |
|---------|---------|--------------|
| `recv_port` | `11000` | ZeroMQ port simulation nodes receive on |
| `send_port` | `11001` | ZeroMQ port simulation nodes send on |

## Paths

| Setting | Default | Description |
|---------|---------|--------------|
| `log_path` | `'output'` | Where data logger output is written |
| `scenario_path` | `'scenario'` | Base directory for resolving scenario filenames |
| `gfx_path` | `'graphics'` | Graphics/resource data path |
| `cache_path` | `'cache'` | Cache data path |
| `navdata_path` | `'navdata'` | Navigation database path |
| `perf_path` | `'performance'` | Aircraft performance data path |
| `perf_path_bada` | `'performance/BADA'` | BADA performance data path (leave empty if you don't have a BADA license) |
| `plugin_path` | `'plugins'` | Directory BlueSky scans for plugins |

## Simulation

| Setting | Default | Description |
|---------|---------|--------------|
| `performance_model` | `'openap'` | Aircraft performance model: `'openap'`, `'bada'`, or `'legacy'` |
| `simdt` | `0.05` | Base simulation time step, seconds |
| `performance_dt` | `1.0` | Performance-model update interval, seconds |
| `fms_dt` | `1.0` | FMS update interval, seconds |
| `prefer_compiled` | `True` | Prefer the compiled C++ geo/ASAS modules over pure-Python fallbacks |
| `max_nnodes` | `999` | Maximum number of simulation nodes a server will spawn |
| `verbose` | `False` | Enable verbose internal logging |
| `enabled_plugins` | `['area', 'datafeed']` | Plugins loaded automatically on startup |
| `start_location` | `'EHAM'` | Where the radar view starts panned/zoomed to |

## Conflict detection & resolution (ASAS)

| Setting | Default | Description |
|---------|---------|--------------|
| `asas_dtlookahead` | `300.0` | Conflict detection lookahead time, seconds |
| `asas_dt` | `1.0` | Conflict detection/resolution update interval, seconds |
| `asas_pzr` | `5.0` | Horizontal protected-zone radius, nautical miles |
| `asas_pzh` | `1000.0` | Vertical protected-zone half-height, feet |
| `asas_marh` | `1.05` | Horizontal resolution margin factor |
| `asas_marv` | `1.05` | Vertical resolution margin factor |

## Qt GUI display

| Setting | Default | Description |
|---------|---------|--------------|
| `text_size` | `13` | Radar screen font size, pixels |
| `apt_size` | `10` | Airport symbol size, pixels |
| `wpt_size` | `10` | Waypoint symbol size, pixels |
| `ac_size` | `16` | Aircraft symbol size, pixels |
| `stack_text_color` | `0, 255, 0` | Console text color (RGB) |
| `stack_background_color` | `102, 102, 102` | Console background color (RGB) |

```{note}
Most of these correspond directly to a command that changes the same value
at runtime (e.g. `DTLOOK` for `asas_dtlookahead`, `IMPL PERFBASE` for
`performance_model`) — see the [command reference](commands/index.md) for
the full set. Modules and plugins can also register their own settings with
`settings.set_variable_defaults(...)`, so this list only covers what ships
with BlueSky's own `default.cfg`.
```
