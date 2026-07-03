# Contributing

BlueSky describes itself as "perpetual beta" — contributions are welcome,
whether that's a bug report, a new plugin, or an improvement to these docs.

## Development install

```console
$ git clone https://github.com/TUDelft-CNS-ATM/bluesky.git
$ cd bluesky
$ pip install -e ".[full]"
```

See [Installation](installation.md) for the C++ compiler requirement (two
small extension modules are built as part of the install).

## Linting and tests

Continuous integration (`.github/workflows/python-test.yml`) runs `flake8`
against every push, across Python 3.10–3.13:

```console
$ pip install flake8
$ flake8 . --count --select=E9,F63,F7,F82 --ignore=F821 --show-source --statistics
```

The project also has a `pytest` test suite under `bluesky/test/`:

```console
$ pip install pytest
$ pytest bluesky/test/
```

## Building this documentation locally

This site is built with Sphinx from the `docs/` directory:

```console
$ pip install -r docs/requirements.txt
$ pip install -e .
$ make -C docs html
```

Open `docs/_build/html/index.html` in a browser to preview. The [stack
command reference](reference/commands/index.md) is regenerated from the
live command registry on every build (see `docs/_ext/gen_commands.py`), so
it always reflects whatever code you're building against.

```{note}
Building the docs imports BlueSky in detached mode (no GUI, no networking)
to read the command registry, so a full install (including the C++
extensions) is required even just to build the docs.
```

## Submitting changes

Please send suggestions, bug fixes, or new features as GitHub pull requests
against [TUDelft-CNS-ATM/bluesky](https://github.com/TUDelft-CNS-ATM/bluesky),
ideally tested and with runtime performance in mind — much of BlueSky's core
is vectorized (see [The traffic model](concepts/traffic-model.md)), so
changes to per-aircraft logic should stay array-based rather than looping
over aircraft in Python where avoidable.

## Getting help

- [Discord](https://discord.gg/wkBKgXCHYN)
- [GitHub discussions](https://github.com/TUDelft-CNS-ATM/bluesky/discussions)
- [GitHub issues](https://github.com/TUDelft-CNS-ATM/bluesky/issues)
