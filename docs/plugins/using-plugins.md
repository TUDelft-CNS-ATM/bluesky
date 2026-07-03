# Using plugins

Plugins are BlueSky's extension mechanism — self-contained Python files that
add stack commands, periodic behaviour, or new/replaced simulation logic
without touching BlueSky's own source. This page covers **activating**
plugins; see [Writing your first plugin](first-plugin.md) if you want to
write one.

## Where plugins live

BlueSky looks for plugin files in the directory named by the `plugin_path`
setting (default `'plugins'`), resolved relative to your [working
directory](../user-guide/configuration.md), *plus* the plugins that ship
with BlueSky itself under `bluesky/plugins/`. Discovery works by scanning
`.py` files for an `init_plugin()` function (BlueSky parses the file's
syntax tree to find it, without importing the file, so a plugin that isn't
enabled has no import-time side effects at all).

## Three ways to activate a plugin

### 1. In `settings.cfg`

List plugins you always want loaded on startup in the `enabled_plugins`
setting:

```python
enabled_plugins = ['area', 'datafeed', 'mycustomplugin']
```

This is the right place for plugins you use in most sessions. The default
is `enabled_plugins = ['datafeed']`.

### 2. From the console (interactively)

```text
PLUGINS LOAD mycustomplugin
```

(`PLUGINS`, also spelled `PLUGIN` or `PLUG-IN`, with subcommands `LIST` and
`LOAD`/`ENABLE`.) This loads a plugin into the currently running simulation
without restarting BlueSky or editing any config file — handy while
developing a plugin, or for plugins you only need occasionally.

List everything that's currently discovered and their load state with:

```text
PLUGINS LIST
```

### 3. From a scenario file

Since `PLUGINS LOAD` is a regular stack command, it can appear as a
timestamped line in a [scenario file](../user-guide/scenario-files.md),
loading the plugin at a specific point in the run:

```text
00:00:00.00>PLUGINS LOAD mycustomplugin
```

This is useful for scenarios that depend on a specific plugin being active,
so the scenario is self-contained and doesn't rely on the user's
`settings.cfg`.

## Plugin types

A plugin declares itself as `'sim'` or `'gui'` type in the config dict it
returns from `init_plugin()`. Only plugins matching the current process's
role are loaded — a `'gui'` plugin (see [GUI plugins](gui-plugins.md)) is
never loaded into a headless simulation node, and vice versa.
