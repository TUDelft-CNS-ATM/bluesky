# The simulation loop and time

The simulation loop lives in `bluesky/simulation/simulation.py`
(`Simulation.run()`/`update()`/`step()`), and drives everything that happens
every tick.

## The main loop

In networked sim mode, `Simulation.run()` is the outermost loop:

```text
while state != END:
    Timer.update_timers()
    bs.net.update()
    self.update()
    bs.scr.update()
```

`update()` handles real-time pacing (sleeping to match wall-clock time when
not fast-forwarding) and then calls `step()`.

## What happens on every step

`Simulation.step()` is the actual per-tick logic:

1. **Always**: process any pending stack commands (`simstack.process()`) —
   this happens even while paused, so you can still type commands during
   `HOLD`.
2. **If the simulation is in the `OP` state:**
   - run all `preupdate` hooks
   - advance the simulation clock (`simtime.step()`)
   - advance UTC time
   - update traffic (`bs.traf.update()` — see [The traffic
     model](traffic-model.md))
   - run all `update` hooks
   - update the plotter and data logger
3. **If paused (`HOLD`):** run the `hold` hooks instead.

`reset()` (triggered by `RESET` or loading a new scenario with `IC`) resets
all subsystems and runs the `reset` hooks.

## Timed functions and hooks

Rather than every subsystem hand-rolling its own timing logic, BlueSky
provides a `@timed_function` decorator (`bluesky/core/timedfunction.py`)
with four hooks: `preupdate`, `update`, `hold`, and `reset`. A function
registered on the `update`/`preupdate` hooks only actually runs every `dt`
seconds of simulation time (via an internal `Timer`), even though the hook
itself is triggered every tick — this is how, for example, conflict
detection can run at 1-second intervals while the base simulation step is
0.05 seconds. Plugins use exactly the same mechanism for their own periodic
logic — see [Timed functions and signals](../plugins/timed-functions-and-signals.md).

## Simulation time

- `simdt` (default `0.05` seconds) is the base simulation time step.
- `dtmult` is a fast-time multiplier: running at `dtmult=10` advances
  simulated time ten times faster than real time.
- **`DT dt`** changes the base time step.
- **`DTMULT n`** sets the fast-time multiplier.
- **`FF [time]`** fast-forwards, optionally to a specific simulation time,
  running as fast as possible rather than pacing to real time.
- **`REALTIME`** toggles real-time pacing on/off.
- **`BENCHMARK`** and **`BATCH`** run scenarios unattended, typically as fast
  as possible.

Time itself is tracked with `decimal.Decimal` precision
(`bluesky/core/simtime.py`) rather than floats, to avoid the accumulating
rounding errors you'd otherwise get from repeatedly adding a small time step
over a long simulation run.
