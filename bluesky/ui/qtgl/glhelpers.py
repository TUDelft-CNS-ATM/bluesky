"""
BlueSky-QtGL tools       : Tools and objects that are used in the BlueSky-QtGL implementation

Methods:
    load_texture(fname)  : GL-texture load function. Returns id of new texture
    BlueSkyProgram()     : Constructor of a BlueSky shader program object: the main shader object in BlueSky-QtGL


Internal methods and classes:
    create_font_array()
    create_attrib_from_glbuf()
    create_attrib_from_nparray()


Created by    : Joost Ellerbroek
Date          : April 2015

Modification  :
By            :
Date          :
------------------------------------------------------------------
"""
try:
    from PyQt4.QtGui import QImage, QPainter, QColor, QFont, QFontMetrics
except ImportError:
    from PyQt5.QtGui import QImage, QPainter, QColor, QFont, QFontMetrics
import OpenGL.GL as gl
import numpy as np
from ctypes import c_void_p, c_float, c_int, Structure, pointer


def load_texture(fname):
    img = QImage(fname)
    ptr = c_void_p(int(img.constBits()))

    tex_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, img.width(), img.height(), 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, ptr)
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    return tex_id


def update_array_buffer(buf_id, data):
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf_id)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, data.nbytes, data)


class GlobalData(Structure):
    _fields_ = [("wrapdir", c_int), ("wraplon", c_float), ("panlat", c_float), ("panlon", c_float), ("zoom", c_float), ("screen_width", c_int), ("screen_height", c_int), ("vertex_scale_type", c_int)]


class BlueSkyProgram():
    # Static variables
    initialized = False
    globaldata  = GlobalData()

    def __init__(self):
        self.shaders = []

    def __init__(self, vertex_shader, fragment_shader):
        self.shaders = []
        self.compile_shader(vertex_shader, gl.GL_VERTEX_SHADER)
        self.compile_shader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        self.link()
        self.init()

    def compile_shader(self, fname, type):
        """Compile a vertex shader from source."""
        source = open(fname, 'r').read()
        shader = gl.glCreateShader(type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        # check compilation error
        result = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not(result):
            raise RuntimeError(gl.glGetShaderInfoLog(shader))
            gl.glDeleteShader(shader)
            return

        self.shaders.append(shader)

    def link(self):
        """Create a shader program with from compiled shaders."""
        self.program = gl.glCreateProgram()
        for i in range(0, len(self.shaders)):
            gl.glAttachShader(self.program, self.shaders[i])

        gl.glLinkProgram(self.program)
        # check linking error
        result = gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
        if not(result):
            raise RuntimeError(gl.glGetProgramInfoLog(self.program))
            gl.glDeleteProgram(self.program)
            for i in range(0, len(self.shaders)):
                gl.glDeleteShader(self.shaders[i])

        # Clean up
        for i in range(0, len(self.shaders)):
            gl.glDetachShader(self.program, self.shaders[i])
            gl.glDeleteShader(self.shaders[i])

    def use(self):
        gl.glUseProgram(self.program)

    def init(self):
        BlueSkyProgram.static_init()

        # Connect the global uniform buffer to this shader
        idx = gl.glGetUniformBlockIndex(self.program, 'global_data')
        gl.glUniformBlockBinding(self.program, idx, 1)

        # If this is a shader with a texture sampler, initialize it
        loc_sampler = gl.glGetUniformLocation(self.program, 'tex_sampler')
        if loc_sampler is not -1:
            gl.glProgramUniform1i(self.program, loc_sampler, 0)

    @staticmethod
    def static_init():
        if not BlueSkyProgram.initialized:
            # First initialization of global uniform buffer
            BlueSkyProgram.ubo_globaldata = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, BlueSkyProgram.ubo_globaldata)
            gl.glBufferData(gl.GL_UNIFORM_BUFFER, 32, pointer(BlueSkyProgram.globaldata), gl.GL_STREAM_DRAW)
            gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 1, BlueSkyProgram.ubo_globaldata)
            BlueSkyProgram.initialized = True

    @staticmethod
    def set_wrap(wraplon, wrapdir):
        BlueSkyProgram.globaldata.wrapdir = wrapdir
        BlueSkyProgram.globaldata.wraplon = wraplon

    @staticmethod
    def set_pan_and_zoom(panlat, panlon, zoom):
        BlueSkyProgram.globaldata.panlat = panlat
        BlueSkyProgram.globaldata.panlon = panlon
        BlueSkyProgram.globaldata.zoom = zoom

    @staticmethod
    def set_win_width_height(w, h):
        BlueSkyProgram.globaldata.screen_width  = w
        BlueSkyProgram.globaldata.screen_height = h

    @staticmethod
    def enable_wrap(flag=True):
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, BlueSkyProgram.ubo_globaldata)
        if flag:
            gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, 4, pointer(BlueSkyProgram.globaldata))
        else:
            tmp = c_int(0)
            gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, 4, pointer(tmp))

    @staticmethod
    def set_vertex_scale_type(vertex_scale_type):
        BlueSkyProgram.globaldata.vertex_scale_type = vertex_scale_type
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, BlueSkyProgram.ubo_globaldata)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, 32, pointer(BlueSkyProgram.globaldata))

    @staticmethod
    def update_global_uniforms():
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, BlueSkyProgram.ubo_globaldata)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, 32, pointer(BlueSkyProgram.globaldata))


class RenderObject(object):
    # Attribute locations
    attrib_vertex, attrib_texcoords, attrib_lat, attrib_lon, attrib_orientation, attrib_color, attrib_texdepth = range(7)
    bound_vao = -1

    def __init__(self, primitive_type=None, first_vertex=0, vertex_count=0):
        self.vao_id             = gl.glGenVertexArrays(1)
        self.enabled_attributes = []
        self.primitive_type     = primitive_type
        self.first_vertex       = first_vertex
        self.vertex_count       = vertex_count

    def bind_attribute(self, attrib_id, size, data, storagetype=gl.GL_STATIC_DRAW, instance_divisor=0, datatype=gl.GL_FLOAT):
        if RenderObject.bound_vao is not self.vao_id:
            gl.glBindVertexArray(self.vao_id)
            RenderObject.bound_vao = self.vao_id

        self.enabled_attributes.append(attrib_id)

        # If the input is an array create a new GL buffer, otherwise assume the buffer already exists and a buffer ID is passed
        if type(data) is np.ndarray:
            # Get an index to one new buffer in GPU mem, bind it, and copy the array data to it
            buf_id = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf_id)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, storagetype)
        else:
            # Assume that a GLuint is passed which means that the buffer is already in GPU memory
            buf_id = data
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf_id)

        # Assign this buffer to one of the attributes in the shader
        gl.glEnableVertexAttribArray(attrib_id)
        gl.glVertexAttribPointer(attrib_id, size, datatype, False, 0, None)
        # For instanced data, indicate per how many instances we move a step in the buffer (1=per instance)
        if instance_divisor > 0:
            gl.glVertexAttribDivisor(attrib_id, instance_divisor)
        # Clean up
        gl.glDisableVertexAttribArray(attrib_id)

        return buf_id

    def bind_vertex_attribute(self, data, storagetype=gl.GL_STATIC_DRAW):
        self.vertexbuf = self.bind_attribute(RenderObject.attrib_vertex, 2, data, storagetype)
        return self.vertexbuf

    def bind_texcoords_attribute(self, data, size=2, storagetype=gl.GL_STATIC_DRAW):
        self.texcoordsbuf = self.bind_attribute(RenderObject.attrib_texcoords, size, data, storagetype)
        return self.texcoordsbuf

    def bind_color_attribute(self, data, storagetype=gl.GL_STATIC_DRAW, instance_divisor=1):
        self.colorbuf = self.bind_attribute(RenderObject.attrib_color, 3, data, storagetype, instance_divisor)
        return self.colorbuf

    def bind_lat_attribute(self, data, storagetype=gl.GL_STATIC_DRAW, instance_divisor=1):
        self.latbuf = self.bind_attribute(RenderObject.attrib_lat, 1, data, storagetype, instance_divisor)
        return self.latbuf

    def bind_lon_attribute(self, data, storagetype=gl.GL_STATIC_DRAW, instance_divisor=1):
        self.lonbuf = self.bind_attribute(RenderObject.attrib_lon, 1, data, storagetype, instance_divisor)
        return self.lonbuf

    def bind_orientation_attribute(self, data, storagetype=gl.GL_STATIC_DRAW, instance_divisor=1):
        self.orientationbuf =  self.bind_attribute(RenderObject.attrib_orientation, 1, data, storagetype, instance_divisor)
        return self.orientationbuf

    def set_vertex_count(self, count):
        self.vertex_count = count

    def set_first_vertex(self, vertex):
        self.first_vertex = vertex

    def draw(self, primitive_type=None, first_vertex=None, vertex_count=None, n_instances=0, latlon=None, color=None):
        if primitive_type is None:
            primitive_type = self.primitive_type

        if first_vertex is None:
            first_vertex = self.first_vertex

        if vertex_count is None:
            vertex_count = self.vertex_count

        if vertex_count == 0:
            return

        if RenderObject.bound_vao is not self.vao_id:
            gl.glBindVertexArray(self.vao_id)
            RenderObject.bound_vao = self.vao_id
            for attrib in self.enabled_attributes:
                gl.glEnableVertexAttribArray(attrib)

        if latlon is not None:
            gl.glVertexAttrib1f(RenderObject.attrib_lat, latlon[0])
            gl.glVertexAttrib1f(RenderObject.attrib_lon, latlon[1])

        if color is not None:
            gl.glVertexAttrib3f(RenderObject.attrib_color, color[0], color[1], color[2])

        if n_instances > 0:
            gl.glDrawArraysInstanced(primitive_type, first_vertex, vertex_count, n_instances)
        else:
            gl.glDrawArrays(primitive_type, first_vertex, vertex_count)

    @staticmethod
    def unbind_all():
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        RenderObject.bound_vao = -1

    @staticmethod
    def create_empty_buffer(size, target=gl.GL_ARRAY_BUFFER, usage=gl.GL_STATIC_DRAW):
        buf_id = gl.glGenBuffers(1)
        gl.glBindBuffer(target, buf_id)
        gl.glBufferData(target, size, None, usage)
        return buf_id


class TextObject(RenderObject):
    loc_char_size = loc_block_size = -1
    tex_id = -1
    char_ar = -1.0

    def __init__(self):
        super(TextObject, self).__init__()
        if TextObject.tex_id is -1:
            TextObject.create_font_array()

    def bind_texdepth_attribute(self, data, storagetype=gl.GL_STATIC_DRAW, instance_divisor=1):
        return self.bind_attribute(RenderObject.attrib_texdepth, 1, data, storagetype, instance_divisor, gl.GL_UNSIGNED_BYTE)

    def prepare_text_string(self, text_string, text_size=16.0, text_color=(0.0, 1.0, 0.0)):
        vertices, texcoords = [], []
        w, h = text_size, text_size * self.char_ar
        for i in range(len(text_string)):
            c = ord(text_string[i])
            # Two triangles per character
            texcoords += [(0, 0, c), (0, 1, c), (1, 0, c), (1, 0, c), (0, 1, c), (1, 1, c)]
            vertices  += [(i * w, h), (i * w, 0.0), ((i + 1) * w, h), ((i + 1) * w, h), (i * w, 0.0), ((i + 1) * w, 0.0)]

        self.bind_vertex_attribute(np.array(vertices, dtype=np.float32))
        self.bind_texcoords_attribute(np.array(texcoords, dtype=np.float32), size=3)
        self.bind_color_attribute(np.array(text_color, dtype=np.float32))

        self.n_instances = -1
        self.text_size = text_size
        self.textblock_size = (len(text_string), 1)

    def prepare_text_instanced(self, text_array, origin_lat, origin_lon, textblock_size, text_color=None, text_size=16.0, vertex_offset=(0.0, 0.0)):
        w, h = text_size, text_size * self.char_ar
        x, y = vertex_offset
        texcoords = [(0, 0, 32), (0, 1, 32), (1, 0, 32), (1, 0, 32), (0, 1, 32), (1, 1, 32)]
        vertices  = [(x, y + h), (x, y), (x + w, y + h), (x + w, y + h), (x, y), (x + w, y)]
        self.bind_vertex_attribute(np.array(vertices, dtype=np.float32))
        self.bind_texcoords_attribute(np.array(texcoords, dtype=np.float32), size=3)

        self.bind_texdepth_attribute(text_array)
        divisor = textblock_size[0] * textblock_size[1]
        self.bind_lat_attribute(origin_lat, instance_divisor=divisor)
        self.bind_lon_attribute(origin_lon, instance_divisor=divisor)

        if text_color is not None:
            self.bind_color_attribute(text_color, instance_divisor=divisor)

        self.textblock_size = textblock_size
        self.text_size = text_size

    def draw(self, position=None, color=None, n_instances=0):
        if RenderObject.bound_vao is not self.vao_id:
            gl.glBindVertexArray(self.vao_id)
            RenderObject.bound_vao = self.vao_id
            for attrib in self.enabled_attributes:
                gl.glEnableVertexAttribArray(attrib)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, TextObject.tex_id)

        if position is not None:
            gl.glVertexAttrib2f(RenderObject.attrib_position, position[0], position[1])

        if color is not None:
            gl.glVertexAttrib3f(RenderObject.attrib_color, color[0], color[1], color[2])

        gl.glUniform2f(TextObject.loc_char_size, self.text_size, self.text_size*self.char_ar)
        if n_instances > 0:
            gl.glUniform2i(TextObject.loc_block_size, self.textblock_size[0], self.textblock_size[1])
            gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 6, n_instances * self.textblock_size[0] * self.textblock_size[1])
        else:
            gl.glUniform2i(TextObject.loc_block_size, 0, 0)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6 * self.textblock_size[0])

    @staticmethod
    def create_font_array(char_height=62, pixel_margin=1, font_family='Courier', font_weight=50):
        # Load font and get the dimensions of one character (assuming monospaced font)
        f = QFont(font_family)
        f.setPixelSize(char_height)
        f.setWeight(font_weight)
        fm = QFontMetrics(f, QImage())

        char_width = char_height = 0
        char_y = 999

        for i in range(32, 127):
            bb = fm.boundingRect(chr(i))
            char_width = max(char_width, bb.width())
            char_height = max(char_height, bb.height())
            char_y = min(char_y, bb.y())

        imgsize = (char_width + 2 * pixel_margin, char_height + 2 * pixel_margin)
        TextObject.char_ar = float(imgsize[1]) / imgsize[0]

        # init the image and the painter that will draw the characters to each image
        img = QImage(imgsize[0], imgsize[1], QImage.Format_ARGB32)
        ptr = c_void_p(int(img.constBits()))
        painter = QPainter(img)
        painter.setFont(f)
        painter.setPen(QColor(255, 255, 255, 255))

        # Set-up the texture array
        TextObject.tex_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, TextObject.tex_id)
        gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, gl.GL_RGBA8, imgsize[0], imgsize[1], 127 - 32, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        # We're using the ASCII range 32-126; space, uppercase, lower case, numbers, brackets, punctuation marks
        for i in range(32, 127):
            img.fill(0)
            painter.drawText(pixel_margin, pixel_margin - char_y, chr(i))
            gl.glTexSubImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, 0, 0, i - 32, imgsize[0], imgsize[1], 1, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, ptr)

        # We're done, close the painter, and return the texture ID, char width and char height
        painter.end()

    @staticmethod
    def init_shader(program):
        TextObject.loc_char_size = gl.glGetUniformLocation(program.program, 'char_size')
        TextObject.loc_block_size = gl.glGetUniformLocation(program.program, 'block_size')
