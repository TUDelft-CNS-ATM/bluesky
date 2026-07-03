# Replaceable entities

Some BlueSky subsystems — the autopilot, conflict detection/resolution, the
performance model — are designed to be **swapped out entirely** rather than
just extended. A plugin can provide an alternative implementation that
callers pick up transparently, with no changes to the code that uses it.
This is what powers the `IMPL`/`IMPLEMENTATION` command.

## How it works

`Base` (`bluesky/core/base.py`) maintains a process-wide registry,
`replaceables`, mapping a base class name (e.g. `CONFLICTRESOLUTION`) to the
base class itself. Any subclass of a class declared `replaceable=True`
automatically registers itself as an alternative implementation for that
base — you don't call any registration function yourself, just subclass it:

```python
class __init_subclass__(cls, replaceable=True, ...):
    ...  # registers cls as an implementation of its replaceable base
```

`Entity` (used for plugin singletons; see [Writing your first
plugin](first-plugin.md)) builds on this with a **`Proxy`** object: external
code always holds a reference to the proxy, which transparently forwards to
whichever concrete implementation is currently selected — so `bs.traf.cr`
keeps working after you switch conflict-resolution algorithms at runtime,
without needing to be reassigned.

## Selecting an implementation

```text
IMPL
```

lists all replaceable bases and their currently selected implementation.

```text
IMPL CONFLICTRESOLUTION EBY
```

selects a specific implementation for a given base — this is what the
`RESO` command effectively does for conflict resolution specifically (see
[Conflict detection and resolution](../concepts/conflict-detection-resolution.md)).

## Worked example: `eby.py`

`bluesky/plugins/asas/eby.py` replaces the conflict-resolution algorithm by
subclassing the existing `ConflictResolution` base class instead of
`core.Entity`:

```python
from bluesky.traffic.asas import ConflictResolution

def init_plugin():
    config = {
        'plugin_name': 'EBY',
        'plugin_type': 'sim',
    }
    return config


class Eby(ConflictResolution):
    def resolve(self, conf, ownship, intruder):
        ''' Resolve all current conflicts. '''
        ...
        return newtrack, neweascapped, newvs, newvspd
```

Loading this plugin (see [Using plugins](using-plugins.md)) registers `Eby`
as an alternative `ConflictResolution` implementation automatically, simply
because it subclasses `ConflictResolution`. Selecting it with `RESO EBY` (or
`IMPL CONFLICTRESOLUTION EBY`) makes `bs.traf.cr` — and everything that
calls into it — use `Eby.resolve()` instead of the default implementation,
with no other code changes anywhere.

## When to use this pattern

Use a replaceable subclass, rather than a plain `core.Entity`, when your
plugin's purpose is to provide an **alternative algorithm for something
BlueSky already does** (a different resolution method, a different
performance model) rather than to **add new, independent behaviour**. If
you're not sure which fits, [Writing your first plugin](first-plugin.md)
covers the additive case, which is the more common starting point.
