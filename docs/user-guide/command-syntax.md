# Stack command syntax

Every BlueSky stack command follows the same basic grammar, whether typed
into the console, played from a [scenario file](scenario-files.md), or
issued by a plugin:

```text
COMMAND arg1,arg2,arg3
```

- The command name is case-insensitive and is upper-cased internally.
- Arguments are separated by commas or spaces.
- Optional arguments are simply omitted from the end (or, for some commands,
  from the middle — see the per-command usage line in the
  [command reference](../reference/commands/index.md)).

For example:

```text
CRE KL204,B738,52.3,4.76,90,FL200,280
ALT KL204,FL100
HDG KL204,270
```

## Argument types

Each command parameter has a **type**, which determines how the typed text
is parsed and validated. These are defined in `bluesky/stack/argparser.py`:

| Type | Meaning |
|------|---------|
| `txt` | A word, upper-cased (e.g. a keyword like `ON`/`OFF`, a shape name) |
| `word` | A single word, case preserved |
| `string` | The rest of the line, taken literally (e.g. free text for `ECHO`) |
| `float` | A floating point number |
| `int` | An integer |
| `onoff` / `bool` | `ON`/`OFF`, `TRUE`/`FALSE`, `1`/`0` |
| `acid` | An aircraft callsign, resolved to its internal index |
| `wpt` | A waypoint or airport identifier |
| `wpinroute` | A waypoint that must already exist in an aircraft's route |
| `latlon` / `lat` | A latitude/longitude pair (or a named fix/airport) |
| `pandir` | A pan direction (`LEFT`/`RIGHT`/`UP`/`DOWN`, or a location) |
| `spd` | A speed, in knots (CAS) or Mach |
| `vspd` | A vertical speed |
| `alt` | An altitude or flight level (e.g. `FL200`, `5000`) |
| `hdg` | A heading in degrees |
| `time` | A duration or time, e.g. `1:30`, `90` |
| `colour` / `color` | A colour name or RGB triplet |

Argument types are attached to a command's parameters either through Python
type annotations on the callback function (`acid: 'acid'`) or through an
`annotations='acid,alt,[spd]'` keyword on the `@stack.command` decorator —
see [Adding stack commands](../plugins/stack-commands.md) if you're writing
your own.

## Optional and repeating arguments

In a command's usage line (as shown in the reference), square brackets mark
optional arguments and `...` marks a repeating (variadic) argument that
consumes the rest of the line:

```text
ADDWPT acid, (wpname/lat,lon), [alt,spd,afterwp]
MOVE acid,lat,lon,[alt,hdg,spd,vspd]
```

Here `alt`, `spd` and `afterwp` may all be omitted from the end of the
`ADDWPT` call.

## Command groups and aliases

Some commands are actually **command groups** with subcommands (e.g.
`ASAS ON`/`ASAS OFF` route to different implementations behind one name),
and many commands have one or more aliases (e.g. `DIRECT` / `DIRECTTO` /
`DIRTO` / `DCT` all do the same thing). Aliases are listed on each command's
[reference page](../reference/commands/index.md).

## Getting help

Type `HELP` for general help, or `HELP <command>` to see a command's usage
and description — the same text shown in this documentation's [command
reference](../reference/commands/index.md). The GUI's doc window shows this
help automatically for whatever command you're currently typing.

## Implicit `POS`

If the first word on the command line isn't a recognized command but *is* an
existing aircraft callsign, BlueSky treats it as `POS <acid>` and shows that
aircraft's status. Double-clicking an aircraft on the radar does the same
thing.
