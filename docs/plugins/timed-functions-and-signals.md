# Timed functions and signals

Plugins hook into BlueSky's simulation loop in two ways: periodic
**timed functions**, and event-driven **signals**.

## Timed functions

`@core.timed_function` (`bluesky/core/timedfunction.py`) registers a method
to run periodically, or on a specific simulation lifecycle event:

```python
@core.timed_function(name='myplugin', dt=5)
def update(self):
    ''' Runs every 5 simulated seconds. '''
    ...
```

Arguments:

- `name` — a unique name for this timed function (used internally to
  deduplicate and to track its own `Timer`).
- `dt` — how often it runs, in simulated seconds. Even though the
  underlying hook fires on every simulation step, an internal `Timer` gates
  the call so your function only actually executes every `dt` seconds —
  regardless of how small the base simulation time step (`simdt`) is.
- `hook` — which lifecycle point to attach to: `'update'` (default),
  `'preupdate'`, `'hold'`, or `'reset'`. `update`/`preupdate` respect `dt`;
  `hold`/`reset` fire once, unconditionally, whenever the simulation is
  paused or reset.

If your function declares a `dt` parameter, it receives the *actual* elapsed
time since it last ran (as a `float`), which can differ slightly from the
requested `dt` depending on the base time step:

```python
@core.timed_function(name='myplugin', dt=1.0)
def update(self, dt):
    # dt ~= 1.0, but exact depending on simdt
    ...
```

This is the same mechanism the [simulation loop](../concepts/simulation-loop.md)
itself uses for the sim's own periodic subsystems (conflict detection,
autopilot, ...) — plugins get identical timing behaviour, not a
lesser-featured variant.

## Signals

`Signal` (`bluesky/core/signal.py`) is a lightweight, named publish/subscribe
mechanism, independent of the simulation clock — used for event-driven
notifications rather than periodic polling. Signal names ("topics") are
case-insensitive and deduplicated process-wide, so subscribing to
`Signal('MYTOPIC')` from two different plugins gets you the same signal
object.

Subscribe with the `@subscriber` decorator:

```python
from bluesky.core import signal

@signal.subscriber(topic='RESET')
def on_reset():
    ''' Called whenever this signal is emitted, anywhere in the process. '''
    ...
```

Or manually:

```python
mysignal = signal.Signal('MYTOPIC')
mysignal.connect(my_callback)
...
mysignal.emit(some_data)   # calls every connected subscriber with some_data
```

BlueSky itself uses signals in several places — for example the network
layer's `@subscriber(topic=...)` decorator (see
[Networking](../concepts/networking.md)) is a thin wrapper for
network-originated messages using this same underlying mechanism.

## Choosing between the two

- Use a **timed function** for anything that should run on a schedule tied
  to simulation time (logging, periodic checks, spawning traffic at
  intervals).
- Use a **signal** for reacting to a specific event, wherever in the
  codebase it originates (reset, a network message arriving, another
  plugin's custom event) — anything that isn't naturally "every N seconds."
