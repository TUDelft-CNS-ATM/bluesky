# BlueSky — The Open Air Traffic Simulator

BlueSky is a tool to perform research on Air Traffic Management and Air Traffic
Flows, distributed under the MIT license.

The goal of BlueSky is to provide everybody who wants to visualize, analyze or
simulate air traffic with a tool to do so without any restrictions, licenses or
limitations. It can be copied, modified, cited, etc. without any limitations.

**Some features of BlueSky:**

- Written in Python 3, using numpy and either Qt+OpenGL or pygame for visualisation
- Extensible by means of self-contained [plugins](plugins/index.md)
- Contains open source data on navaids, performance data of aircraft and geography,
  with global coverage of navaid and airport data
- Simulates aircraft performance, flight management system (LNAV, VNAV),
  autopilot, conflict detection and resolution, and airborne separation assurance
- Compatible with BADA 3.x data
- Traffic is controlled via user inputs in a console window or by playing
  scenario files (`.scn`) containing the same commands with a timestamp
- Mouse clicks in the traffic window can complete lat/lon/heading and position
  arguments of the command being typed

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user-guide/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Simulator concepts

concepts/index
```

```{toctree}
:maxdepth: 2
:caption: Plugins

plugins/index
```

```{toctree}
:maxdepth: 2
:caption: Python API

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
citation
legacy-docs
```

## Questions or suggestions?

Visit the BlueSky community on [Discord](https://discord.gg/wkBKgXCHYN), open a
topic on the [GitHub discussion board](https://github.com/TUDelft-CNS-ATM/bluesky/discussions),
or open an [issue](https://github.com/TUDelft-CNS-ATM/bluesky/issues).
