# PLUGINS — List and load plugins

```text
PLUGINS [LIST/LOAD/ENABLE] [plugin_name]
```

Manages plugins in the running simulation (aliases: `PLUGIN`, `PLUG-IN`,
`PLUG-INS`).

| Argument | Type | Default | Description |
|----------|------|---------|--------------|
| `cmd` | text | `LIST` | `LIST`, `LOAD`, or `ENABLE` |
| `plugin_name` | text | *(none)* | The plugin to load, for `LOAD`/`ENABLE` |

## Subcommands

- **`PLUGINS LIST`** — shows currently running plugins and any additional
  plugins that were discovered but aren't loaded yet.
- **`PLUGINS LOAD name`** (or `PLUGINS ENABLE name`) — loads a plugin into
  the running simulation immediately.
- **`PLUGINS name`** (no explicit `LOAD`) also works — if the first argument
  isn't `LIST`/`LOAD`/`ENABLE`, it's treated as a plugin name to load.

## Examples

```text
PLUGINS LIST
PLUGINS LOAD area
PLUGINS datafeed
```

## Notes

- Only plugins matching the current process's type (`sim` vs `gui`) show up
  here — a headless simulation node never lists `gui`-type plugins, for
  example.
- This is one of three ways to activate a plugin — see [Using
  plugins](../../plugins/using-plugins.md) for the other two
  (`enabled_plugins` in `settings.cfg`, and a `PLUGINS LOAD` line inside a
  scenario file).
