''' BlueSky tiled map GL object. '''
import numpy as np

import bluesky as bs
from bluesky.ui.qtgl import glhelpers as glh
from bluesky.ui.qtgl.glmap import Map
from bluesky.ui.qtgl.tiledtexture import TiledTexture


bs.settings.set_variable_defaults(tilesource='opentopomap')


class TiledMap(Map):
    @classmethod
    def select(cls):
        ''' Echo tile source usage message to console before selecting TiledMap. '''
        tileinfo = bs.settings.tile_sources.get(bs.settings.tilesource)
        if not tileinfo:
            raise KeyError(f'Tile source {bs.settings.tilesource} not found!')
        msg = tileinfo.get('license', f'Tile source: {bs.settings.tilesource}')
        bs.stack.stack(f'ECHO {msg}')
        return super().select()

    ''' Tiled Texture background for radarview. '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.map = glh.VertexArrayObject(glh.gl.GL_TRIANGLE_FAN, shader_type='tiled')
        self.texture = TiledTexture(self.glsurface, bs.settings.tilesource)
        self.offsetzoom_loc = 0

    def create(self):
        mapvertices = np.array(
            [-90.0, 540.0, -90.0, -540.0, 90.0, -540.0, 90.0, 540.0], dtype=np.float32)
        
        self.texture.create()
        self.texture.add_bounding_box(-90, -180, 90, 180)
        self.map.create(vertex=mapvertices, texture=self.texture)
        self.offsetzoom_loc = glh.ShaderSet.get_shader(
            'tiled').uniformLocation('offset_scale')
        # Make sure that we have textures on first draw
        self.texture.on_panzoom_changed(True)

    def draw(self):
        # Send the (possibly) updated global uniforms to the buffer
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)

        # --- DRAW THE MAP AND COASTLINES ---------------------------------------------
        # Map and coastlines: don't wrap around in the shader
        self.shaderset.enable_wrap(False)
        shader = glh.ShaderSet.get_shader('tiled')
        shader.bind()
        shader.setUniformValue(self.offsetzoom_loc, *self.texture.offsetscale)
        self.map.draw()
        # Temp coastlines
        Map._instance.draw(skipmap=True)
