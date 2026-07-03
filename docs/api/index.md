# Using BlueSky as a Python library

BlueSky can be embedded directly in a Python script or notebook, without
starting a GUI or any networking — this is the same **detached mode** used
by `--detached` on the command line, and by this documentation's own
[command-reference generator](../reference/commands/index.md).

## Initializing

```python
import bluesky as bs

bs.init(mode='sim', detached=True)
```

After this call, the usual [global singletons](../concepts/architecture.md)
are populated: `bs.traf`, `bs.sim`, `bs.navdb`, `bs.settings`, etc. No ZMQ
sockets are opened and no GUI is started — `bs.net` is replaced with a no-op
stand-in (see [Networking](../concepts/networking.md)).

Pass an explicit `workdir=` if you don't want BlueSky creating or reusing
`~/bluesky` or the current directory as its working directory:

```python
import tempfile
bs.init(mode='sim', detached=True, workdir=tempfile.mkdtemp())
```

## Driving the simulation

Issue stack commands the same way a scenario file would, and step the
simulation manually:

```python
bs.stack.stack('CRE KL204,B738,52.3,4.76,90,FL200,280')
bs.stack.stack('OP')

for _ in range(1000):
    bs.sim.step()
```

## Reading traffic state

Aircraft state lives in parallel NumPy arrays on `bs.traf` (see [The
traffic model](../concepts/traffic-model.md)):

```python
print(bs.traf.id)     # list of callsigns
print(bs.traf.lat)     # array of latitudes
print(bs.traf.alt)     # array of altitudes, in meters
```

## Typical uses

- Scripting batches of experiments without a GUI
- Programmatically generating or perturbing traffic
- Wrapping BlueSky as an environment for reinforcement learning or other
  automated decision-making research (a common use of BlueSky in the ATM
  research community)
- Automated tooling like this documentation site's command-reference
  generator, which uses exactly this detached-mode pattern to introspect
  the live stack command registry

## API reference

The following pages document the modules most relevant when embedding
BlueSky or writing plugins:

```{toctree}
:maxdepth: 2

core
stack
traffic
tools
```
