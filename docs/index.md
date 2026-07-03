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
:maxdepth: 1
:caption: User guide

user-guide/ui-tour
user-guide/command-syntax
user-guide/scenario-files
user-guide/running-modes
user-guide/configuration
user-guide/logging-and-output
user-guide/other-uis
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/commands/index
reference/cli
reference/settings-reference
```

```{toctree}
:maxdepth: 1
:caption: Simulator concepts

concepts/architecture
concepts/simulation-loop
concepts/traffic-model
concepts/performance-models
concepts/autopilot-fms
concepts/conflict-detection-resolution
concepts/wind-weather-adsb
concepts/networking
```

```{toctree}
:maxdepth: 1
:caption: Plugins

plugins/index
plugins/using-plugins
plugins/first-plugin
plugins/traffic-arrays
plugins/timed-functions-and-signals
plugins/stack-commands
plugins/replaceable-entities
plugins/gui-plugins
plugins/custom-importers
```

```{toctree}
:maxdepth: 1
:caption: Python API

api/index
api/core
api/stack
api/traffic
api/tools
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
