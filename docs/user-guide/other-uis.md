# Other user interfaces

The Qt/OpenGL GUI is BlueSky's primary interface, but two lighter-weight
alternatives exist for cases where a full OpenGL window isn't wanted or
available.

## Pygame GUI

`bluesky/ui/pygame/` provides a legacy pygame-based GUI (`screen.py`,
`console.py`, `menu.py`, `keyboard.py`, `dialog.py`). It offers the same
basic radar-and-console experience with fewer dependencies than Qt/OpenGL.
Install it with the `pygame` extra:

```console
$ pip install "bluesky-simulator[pygame]"
```

and start it as a networked client (`--client`) with the pygame GUI, or via
`BlueSky_pygame.py` from a source checkout.

## Text console client

`bluesky/ui/console/` provides a terminal-based client (built on
[Textual](https://textual.textualize.io/)), useful for interacting with a
running server over SSH or in environments without any display at all.
Install the `console` extra and connect with:

```console
$ pip install "bluesky-simulator[console]"
$ bluesky --console [hostname]
```

Without a hostname, a discovery dialog lets you pick from servers found on
the local network (if the server was started with `--discoverable`).

## No web UI

BlueSky does not currently ship a browser-based UI. The HTML files under
`bluesky/resources/html/` are static documentation pages shown inside the
Qt GUI's doc window, not a web interface to the simulator.
