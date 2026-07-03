# A tour of the BlueSky user interface

BlueSky's primary interface is the **Qt/OpenGL GUI**
(`bluesky/ui/qtgl/`). It combines a radar view, a command console, and a
handful of secondary windows.

## Window layout

- **Radar view** (`radarwidget.py`) — the main OpenGL view of traffic, map,
  navigation data and drawn shapes. Composited from several layers: traffic
  symbols, the base map, navigation data, polygons/areas, and (optionally)
  tiled background maps.
- **Console** (`console.py`) — the command line and scrollback at the bottom
  of the window, where you type stack commands and see their output.
- **Navigation display (ND)**, **arrival manager**, **traffic list**,
  **stack/command history list**, **area list**, **info window**, **settings
  window** and **doc window** — secondary windows/panels reachable from the
  menu, each showing a different slice of simulation state.

## The console

- **Enter/Return** — submit the typed command to the simulation
- **Tab** — autocomplete the current command or argument
- **Up / Down** — step through command history
- **Left / Right** — move the cursor
- **Ctrl+V** — paste (single line only)

Anything you type that isn't a recognized keyboard shortcut goes straight
into the console's command line — you don't need to click into it first.

## Keyboard shortcuts

| Key | Effect |
|-----|--------|
| Shift + Arrow keys | Pan the radar view |
| F11 | Toggle full-screen |
| Esc | Close / quit |
| *(any other key)* | Forwarded to the console command line |

## Mouse and trackpad

| Gesture | Effect |
|---------|--------|
| Left-drag | Pan the radar view |
| Left-click (no drag) | Emits a click event with the clicked lat/lon — see below |
| Double-click on an aircraft | Issues `POS <acid>` for that aircraft |
| Ctrl/Cmd + scroll wheel | Zoom, centered on the cursor |
| Two-finger trackpad scroll | Pan |
| Pinch gesture | Zoom |

## Radar-click command completion

This is one of BlueSky's more distinctive UI features: **clicking the radar
view can fill in the argument you're currently typing**, based on what
command is on the command line (`bluesky/ui/radarclick.py`).

For example:

- Type `CRE KL204,B738` and click a point on the map — the click fills in
  `lat,lon` for the new aircraft's position.
- Type `ADDWPT KL204` and click near a waypoint or airport — the click
  resolves to the nearest matching waypoint name.
- Type `DEST KL204` and click near an airport — the click fills in that
  airport as the destination.
- Double-clicking directly on an aircraft symbol issues `POS <acid>` for
  that aircraft, showing its full status.

The exact behaviour depends on which argument type the currently-typed
command expects next (aircraft id, lat/lon, heading, distance, waypoint,
...) — see [Stack command syntax](command-syntax.md) for the list of
argument types.

## Other windows

- **Doc window** — shows the built-in HTML documentation for the command
  currently being typed (the same per-command pages this site's [command
  deep dives](../reference/commands/index.md) are partly seeded from).
- **Info window** — simulation and node status.
- **Settings window** — edit settings interactively (see
  [Configuration](configuration.md)).

## Other user interfaces

The Qt GUI isn't the only way to interact with BlueSky — see
[Other user interfaces](other-uis.md) for the pygame GUI and the text-based
console client, useful for headless servers or lightweight setups.
