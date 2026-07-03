# Custom scenario importers

BlueSky normally loads traffic and commands from
[`.scn` scenario files](../user-guide/scenario-files.md). If you have
existing flight data or traffic logs in some other format, you can teach
BlueSky to read it directly with the `Importer` extension point
(`bluesky/stack/importer.py`), instead of writing a one-off conversion
script.

## The `Importer` interface

`bluesky/plugins/importer_example.py` shows the pattern:

```python
from bluesky.stack.importer import Importer

def init_plugin():
    config = {
        'plugin_name': 'IMPORTEX',
        'plugin_type': 'sim',
    }
    return config


class ImportExample(Importer):
    def __init__(self):
        super().__init__(filetype='Example', extensions=('txt', 'dat', 'log'))

    def load(self, fname):
        # Return two lists:
        # - timestamps (in seconds, or fractions of a second); pass an
        #   empty list if the commands aren't timestamped
        # - the corresponding stack commands
        return [], ['MCRE 1']
```

- `filetype` is a human-readable label for this importer (shown in file
  dialogs).
- `extensions` lists the file extensions this importer should handle.
- `load(fname)` does the actual parsing and returns `(timestamps,
  commands)` — the same shape of data BlueSky gets from parsing a `.scn`
  file, just produced from your own format instead.

Once registered (by loading the plugin — see [Using
plugins](using-plugins.md)), files with a matching extension can be loaded
the same way a `.scn` file would be, e.g. via `IC` or a file dialog.

## When to use this

Use a custom importer when you have a recurring, structured data source
(e.g. exported flight plans, a proprietary traffic log format, output from
another simulator) that you'd otherwise be hand-converting to `.scn` syntax
every time. For a one-off conversion, it's often simpler to just write a
small script that generates a `.scn` file directly — reserve a custom
`Importer` for formats you load repeatedly.
