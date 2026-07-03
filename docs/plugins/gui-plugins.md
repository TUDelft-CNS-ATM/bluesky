# GUI plugins

Most plugins are `'sim'`-type: they run inside the simulation node and touch
traffic state, commands, and timing. A plugin can instead be `'gui'`-type,
running inside the Qt GUI client and drawing on the radar view.

## Declaring a GUI plugin

```python
def init_plugin():
    config = {
        'plugin_name': 'GUIEXAMPLE',
        'plugin_type': 'gui',
    }
    return config
```

Only `'gui'`-type plugins are loaded into GUI client processes, and only
`'sim'`-type plugins are loaded into simulation nodes — a `'gui'` plugin is
never loaded into a headless server, and vice versa (see [Using
plugins](using-plugins.md)).

## Drawing on the radar view

`bluesky/plugins/guiexample.py` shows the pattern: subclass `RenderObject`
(`bluesky/ui/qtgl/glhelpers.py`) instead of `core.Entity`, specifying a
render `layer`:

```python
from bluesky import stack
from bluesky.ui.qtgl.glhelpers import gl, RenderObject, VertexArrayObject

class MyVisual(RenderObject, layer=100):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.shape = VertexArrayObject(gl.GL_TRIANGLE_FAN)

    def create(self):
        vertices = np.array([52, 5, 52, 4, 54, 4, 54, 5], dtype=np.float32)
        self.shape.create(vertex=vertices, color=(255, 0, 0))

    def draw(self):
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)
        self.shape.draw()

    @stack.command
    def mycolor(self, *colour: 'colour'):
        ''' Set the color of my example shape. '''
        self.shape.single_color[1:4] = colour
```

- `create()` builds the OpenGL geometry once (here: a red quad defined by
  lat/lon vertices).
- `draw()` runs every frame; `set_vertex_scale_type(VERTEX_IS_LATLON)` tells
  the shader the vertex coordinates are geographic rather than screen or
  meter-based (the other options are `VERTEX_IS_METERS` and
  `VERTEX_IS_SCREEN`, for shapes anchored to a real-world size or to a fixed
  screen position, respectively).
- `layer=100` controls draw order relative to other layers (traffic, map,
  navdata, etc. — higher draws on top).

## Stack commands from a GUI plugin

GUI plugins can register stack commands exactly like `'sim'` plugins do
(the `@stack.command` decorator works identically) — here `MYCOLOR` takes a
variadic `colour` argument (three RGB integers, via the starred `*colour:
'colour'` annotation), and updates the shape's color at runtime. Note that
commands registered from a GUI plugin only affect the client's own display —
they don't reach the simulation side unless you explicitly forward them.
