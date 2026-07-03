# Writing your first plugin

BlueSky ships a template plugin at `bluesky/plugins/example.py` that's the
best starting point for a new plugin. This page walks through it line by
line, then shows how to turn it into your own plugin.

## The `init_plugin()` contract

Every plugin file must define a function called exactly `init_plugin()`.
BlueSky's plugin scanner (`bluesky/core/plugin.py`) statically parses each
`.py` file in the plugin search path looking for this function, without
importing files it isn't going to load — so having many plugin files around
that aren't enabled costs nothing.

`init_plugin()` must return either:

- a single **config dict**, or
- a `(config, stackfunctions)` tuple, where `stackfunctions` is a legacy-style
  dict of additional stack commands (modern plugins register commands with
  the `@stack.command` decorator instead — see [Adding stack
  commands](stack-commands.md)).

The config dict needs at least:

```python
config = {
    'plugin_name': 'EXAMPLE',   # unique name, upper-cased internally
    'plugin_type': 'sim',       # 'sim' or 'gui'
}
```

For `'sim'`-type plugins using the older, function-based style, the config
dict can also carry `update_interval`, and callables `preupdate`, `update`,
`reset`, which get wired up as timed functions automatically. The modern
style (used in `example.py` and recommended for new plugins) instead defines
an `Entity` subclass with `@core.timed_function` and `@stack.command`
decorators, and `init_plugin()` just instantiates it and returns the config.

## Walking through `example.py`

```python
from random import randint
import numpy as np
from bluesky import core, stack, traf

def init_plugin():
    example = Example()
    config = {
        'plugin_name':     'EXAMPLE',
        'plugin_type':     'sim',
    }
    return config


class Example(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.npassengers = np.array([])

    def create(self, n=1):
        super().create(n)
        self.npassengers[-n:] = [randint(0, 150) for _ in range(n)]

    @core.timed_function(name='example', dt=5)
    def update(self):
        stack.stack('ECHO Example update: creating a random aircraft')
        stack.stack('MCRE 1')

    @stack.command
    def passengers(self, acid: 'acid', count: int = -1):
        ''' Set the number of passengers on aircraft 'acid' to 'count'. '''
        if count < 0:
            return True, f'Aircraft {traf.id[acid]} currently has {self.npassengers[acid]} passengers on board.'
        self.npassengers[acid] = count
        return True, f'The number of passengers on board {traf.id[acid]} is set to {count}.'
```

Piece by piece:

- `init_plugin()` instantiates `Example()` *once* — `Entity` subclasses are
  singletons (see [Replaceable entities](replaceable-entities.md)) — and
  returns the config dict identifying this as a `sim`-type plugin named
  `EXAMPLE`.
- `Example.__init__` registers a per-aircraft array, `npassengers`, using
  `self.settrafarrays()`. Because it's registered this way, the array is
  automatically resized whenever aircraft are created or deleted — see
  [Traffic arrays](traffic-arrays.md).
- `create(self, n=1)` is called automatically whenever `n` new aircraft are
  created; it must call `super().create(n)` first, then fill in the new
  slots of any custom arrays.
- `update()` is decorated `@core.timed_function(name='example', dt=5)`, so it
  runs every 5 simulated seconds regardless of the base simulation time
  step — see [Timed functions and signals](timed-functions-and-signals.md).
  It uses `stack.stack(...)` to inject stack commands as if they'd been
  typed by the user.
- `passengers(...)` is decorated `@stack.command`, turning it into the
  `PASSENGERS acid, [count]` stack command. The `acid: 'acid'` annotation
  converts a typed callsign into the aircraft's array index automatically —
  see [Adding stack commands](stack-commands.md) for the full annotation
  syntax.

## Making it your own

1. Copy `example.py` to a new file in your plugin directory (see [Using
   plugins](using-plugins.md) for where that is), and rename the class and
   `plugin_name`.
2. Replace `npassengers` with whatever per-aircraft state you need, or
   remove `settrafarrays()`/`create()` entirely if your plugin doesn't need
   per-aircraft state.
3. Replace the `update()` body with your own periodic logic, and adjust
   `dt` to how often it should run.
4. Add whatever stack commands make sense for your plugin with
   `@stack.command`.
5. Load it with `PLUGINS LOAD <name>` to test it interactively, then add it
   to `enabled_plugins` once it's working — see [Using
   plugins](using-plugins.md).

## Where to go next

- [Traffic arrays](traffic-arrays.md) — per-aircraft state in depth
- [Timed functions and signals](timed-functions-and-signals.md) — periodic
  logic and the publish/subscribe `Signal` system
- [Adding stack commands](stack-commands.md) — the full argument-annotation
  syntax
- [Replaceable entities](replaceable-entities.md) — *replacing* existing
  BlueSky logic (e.g. conflict resolution) instead of adding new logic
- [GUI plugins](gui-plugins.md) — drawing on the radar view instead of
  simulation logic
