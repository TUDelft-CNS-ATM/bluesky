# Installation

BlueSky requires Python 3.10 or newer, and runs on Windows, macOS and Linux.

## Installing with pip

BlueSky is released on PyPI as [`bluesky-simulator`](https://pypi.python.org/pypi/bluesky-simulator/).
The recommended install for regular use is the `full` variant, which includes
the Qt GUI:

```console
$ pip install "bluesky-simulator[full]"
```

The quotes around the package name are required by some shells (e.g. ZSH).

### Optional extras

The pip package comes in several variants, so you only install the
dependencies you need:

| Extra        | Installs                                             | Use when                                    |
|--------------|------------------------------------------------------|---------------------------------------------|
| `full`       | Everything below                                     | You want the complete experience             |
| `qt6`        | PyQt6, PyOpenGL, `bluesky-guidata`                   | You want the standard Qt/OpenGL GUI          |
| `console`    | textual                                              | You want the text-based console client       |
| `pygame`     | pygame                                               | You want the legacy pygame GUI               |
| `headless`   | *(no extra dependencies)*                            | Running server-only, e.g. batch experiments  |

For a headless (server-only) installation, install the bare package:

```console
$ pip install bluesky-simulator
```

Navigation data (world-wide navaids, airports, and airways) is installed
automatically through the companion package `bluesky-navdata`. GUI graphics
come from `bluesky-guidata`, pulled in by the `qt6` and `full` extras.

After installation the `bluesky` command is available:

```console
$ bluesky            # start server + Qt GUI
$ bluesky --help     # list all command line options
```

When run as a pip package, BlueSky creates a working directory at `~/bluesky`
with folders for your scenarios, plugins, output and cache, and a
`settings.cfg` file. See [Configuration](user-guide/configuration.md).

## Installing from source

Clone the repository and install in editable mode:

```console
$ git clone https://github.com/TUDelft-CNS-ATM/bluesky.git
$ cd bluesky
$ pip install -e .
```

Add the extras you need, e.g. `pip install -e ".[full]"`.

BlueSky contains two small C++ extension modules (for geo calculations and
conflict detection). These are compiled during installation, which requires a
C++ compiler to be available:

- **Linux**: `gcc`/`g++` (e.g. `sudo apt install build-essential`)
- **macOS**: the Xcode command line tools (`xcode-select --install`)
- **Windows**: the Microsoft C++ Build Tools

From a source checkout, BlueSky uses the repository itself as working
directory: scenarios are read from `scenario/`, plugins from `plugins/`, and
output goes to `output/`.

To start BlueSky from a source checkout:

```console
$ python BlueSky.py
```

## Aircraft performance models

BlueSky ships with the open-source [OpenAP](https://github.com/junzis/openap)
aircraft performance model, which is used by default. It also supports
EUROCONTROL's BADA 3.x model. **BADA data is license-restricted and not
distributed with BlueSky**: if you have a BADA license, place the data files
in the `performance/BADA` directory of your BlueSky resources, and select the
model with the `IMPL` stack command or the `performance_model` setting. See
[Performance models](concepts/performance-models.md).
