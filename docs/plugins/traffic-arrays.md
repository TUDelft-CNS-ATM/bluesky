# Traffic arrays

BlueSky stores aircraft state as parallel NumPy arrays and Python lists
rather than one object per aircraft (see [The traffic
model](../concepts/traffic-model.md)). `TrafficArrays`
(`bluesky/core/trafficarrays.py`) is the mechanism that lets your own
plugin's per-aircraft state stay in sync with the rest of the simulation —
resized automatically whenever aircraft are created or deleted — without
writing that bookkeeping yourself.

`Entity` (used for all plugin singletons — see [Writing your first
plugin](first-plugin.md)) already inherits from `TrafficArrays`, so any
`core.Entity` subclass gets this for free.

## Registering arrays

Register per-aircraft arrays or lists inside a `with self.settrafarrays():`
block, typically in `__init__`:

```python
class MyPlugin(core.Entity):
    def __init__(self):
        super().__init__()
        with self.settrafarrays():
            self.fuelused = np.array([])      # numpy array -> per-aircraft float
            self.callsign_log = []            # plain list -> per-aircraft string/object
```

Anything assigned inside the `with` block that is a `list` or `np.ndarray`
is picked up automatically (by comparing the object's `__dict__` before and
after the block) and registered for auto-resizing.

## Keeping arrays in sync

Two methods, both inherited from `TrafficArrays`, do the bookkeeping:

- **`create(self, n=1)`** — called automatically whenever `n` new aircraft
  are created. The base implementation appends `n` default values (`0.0`
  for floats, `''` for strings, etc. — see the `defaults` dict in
  `trafficarrays.py`) to every registered array/list. If your plugin needs
  different initial values than the type default, override `create()`,
  **always calling `super().create(n)` first**, then set the values for the
  new elements (the last `n` positions):

  ```python
  def create(self, n=1):
      super().create(n)
      self.fuelused[-n:] = 0.0
  ```

- **`delete(self, idx)`** / **`reset(self)`** — remove an aircraft's data
  from every registered array, or clear everything back to zero aircraft.
  You don't normally need to override these; the base implementation
  already removes the corresponding element/index from every registered
  array or list.

## Nesting

`TrafficArrays` objects form a tree rooted at `bs.traf` itself
(`TrafficArrays.setroot(...)`). If a `TrafficArrays`-derived object is
itself assigned as an attribute inside a `settrafarrays()` block, it's
automatically reparented into the tree, so `create`/`delete`/`reset` cascade
through nested sub-entities (this is how, for example, `Traffic`'s
sub-entities like `Autopilot` and `Route` all stay in sync without each one
re-implementing the same bookkeeping).

## A note on plugins loaded mid-simulation

If your plugin's entity is instantiated while aircraft already exist (e.g.
you `PLUGINS LOAD` it after the simulation has already created aircraft),
`_init_trafarrays` detects the existing aircraft count and calls `create()`
immediately to backfill your new arrays to the correct size — you don't
need to handle this case specially.
