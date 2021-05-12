''' BlueSky OpenGL map object. '''
from os import path
import numpy as np

from bluesky.ui import palette
import bluesky.ui.qtgl.glhelpers as glh
from bluesky.navdatabase import load_aptsurface, load_coastlines
from bluesky import settings


settings.set_variable_defaults(gfx_path='data/graphics')
palette.set_default_colours(
    background=(0, 0, 0),
    coastlines=(85, 85, 115))


class Map(glh.RenderObject):
    ''' Radar screen map OpenGL object. '''
    def __init__(self, parent):
        super().__init__(parent)

        self.map = glh.VertexArrayObject(glh.gl.GL_TRIANGLE_FAN)
        self.coastlines = glh.VertexArrayObject(glh.gl.GL_LINES)
        self.coastindices = []
        self.vcount_coast = 0
        self.wraplon_loc = 0

    def create(self):
        ''' Create GL objects. '''
        # ------- Coastlines -----------------------------
        coastvertices, self.coastindices = load_coastlines()
        self.coastlines.create(vertex=coastvertices, color=palette.coastlines)
        self.vcount_coast = len(coastvertices)

        mapvertices = np.array(
            [-90.0, 540.0, -90.0, -540.0, 90.0, -540.0, 90.0, 540.0], dtype=np.float32)
        texcoords = np.array(
            [1, 3, 1, 0, 0, 0, 0, 3], dtype=np.float32)
        self.wraplon_loc = glh.ShaderSet.get_shader(self.coastlines.shader_type).attribs['lon'].loc

        # Load and bind world texture
        max_texture_size = glh.gl.glGetIntegerv(glh.gl.GL_MAX_TEXTURE_SIZE)
        print('Maximum supported texture size: %d' % max_texture_size)
        for i in [16384, 8192, 4096]:
            if max_texture_size >= i:
                fname = path.join(settings.gfx_path,
                                  'world.%dx%d.dds' % (i, i // 2))
                print('Loading texture ' + fname)
                self.map.create(vertex=mapvertices,
                                texcoords=texcoords, texture=fname)
                break

    def draw(self):
        self.map.draw()
        shaderset = glh.ShaderSet.selected
        if shaderset.data.wrapdir == 0:
            # Normal case, no wrap around
            self.coastlines.draw(
                first_vertex=0, vertex_count=self.vcount_coast)
        else:
            self.coastlines.bind()
            shader = glh.ShaderProgram.bound_shader
            wrapindex = np.uint32(
                self.coastindices[int(shaderset.data.wraplon) + 180])
            if shaderset.data.wrapdir == 1:
                shader.setAttributeValue(self.wraplon_loc, 360.0)
                self.coastlines.draw(
                    first_vertex=0, vertex_count=wrapindex)
                shader.setAttributeValue(self.wraplon_loc, 0.0)
                self.coastlines.draw(
                    first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex)
            else:
                shader.setAttributeValue(self.wraplon_loc, -360.0)
                self.coastlines.draw(
                    first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex)
                shader.setAttributeValue(self.wraplon_loc, 0.0)
                self.coastlines.draw(
                    first_vertex=0, vertex_count=wrapindex)