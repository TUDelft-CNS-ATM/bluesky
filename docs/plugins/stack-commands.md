# Adding stack commands

Plugins add new [stack commands](../user-guide/command-syntax.md) the same
way BlueSky's own modules do: with the `@stack.command` decorator
(`bluesky/stack/cmdparser.py`).

## Basic usage

```python
from bluesky import stack

@stack.command
def myaction(acid: 'acid', count: int = 5):
    ''' This docstring becomes the command's help text. '''
    ...
    return True, f'Done for {count} aircraft'
```

- The command name defaults to the function name, upper-cased (`MYACTION`
  here). Override it with `@stack.command(name='SOMETHINGELSE')`.
- Each parameter's type comes from its **type annotation** — either a
  BlueSky-specific string type like `'acid'` or `'latlon'` (see the
  argument type table in [Stack command syntax](../user-guide/command-syntax.md)),
  or a plain Python type (`int`, `float`, `str`). A parameter with a
  default value is optional.
- The docstring becomes the command's `help` text (shown by `HELP
  MYACTION`); a `brief` and `aliases` can also be passed as decorator
  keyword arguments if you want to override the auto-generated usage line.
- Return `(success, message)` — a bool and a string shown to the user. If
  you return nothing, BlueSky treats the call as successful with an empty
  message.

For methods on an `Entity` (the usual case for plugins — see [Writing your
first plugin](first-plugin.md)), just decorate the method; `self` is handled
automatically:

```python
class MyPlugin(core.Entity):
    @stack.command
    def passengers(self, acid: 'acid', count: int = -1):
        ...
```

## The `annotations` keyword

Instead of (or in addition to) per-parameter type annotations, you can pass
a compact `annotations=` string, matching the same argument-type vocabulary
used throughout BlueSky's own command table (e.g.
`'acid,alt,[spd]'` in `ALT acid, alt, [vspd]`):

- Plain names (`acid`, `alt`, `txt`, ...) map to argument types.
- `[...]` around a name makes it optional.
- `...` at the end means the last parameter gobbles the remainder of the
  line (only valid if the function's last parameter is `*args`-style).

## Command groups: subcommands and alternatives

`@stack.commandgroup` produces a `CommandGroup`, which supports:

- **Subcommands**, added with `@yourcommand.subcommand`: when the first word
  after the command name matches a registered subcommand, that subcommand's
  callback runs instead of the group's own.
- **Alternative implementations**, added with `@yourcommand.altcommand`: if
  the group's own callback reports failure, each registered alternative is
  tried in turn until one succeeds. This is how the same command name can
  be extended by unrelated plugins without conflicting — e.g. a resolution
  method plugin can add itself as an alternative under a shared command
  name rather than needing a brand-new one.

## Legacy dict-based registration

Older code (and the built-in `bluesky/stack/basecmds.py`) registers many
commands at once via a plain dict passed to `append_commands()`, mapping
name → `(brief, annotations, function, help)`. A plugin's `init_plugin()`
can still return a `(config, stackfunctions)` tuple using this same format
if you're porting an older plugin, but for new plugins the `@stack.command`
decorator on an `Entity` method (as in `example.py`) is simpler and is what
this documentation recommends.

## Everything converges on one registry

Regardless of which style registers a command, they all end up in the same
place: `bluesky.stack.cmdparser.Command.cmddict`. This is also what the
[auto-generated command reference](../reference/commands/index.md) on this
site reads from — so a plugin's commands, once loaded, show up in exactly
the same kind of listing as BlueSky's built-in ones.
