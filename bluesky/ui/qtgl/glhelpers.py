''' BlueSky OpenGL classes and functions. '''
import importlib
from os import path
from collections import namedtuple
from collections import OrderedDict
from pathlib import Path

try:
    from PyQt5.QtCore import qCritical, QT_VERSION_STR
    from PyQt5.QtWidgets import QOpenGLWidget
    from PyQt5.QtGui import (QSurfaceFormat, QOpenGLShader, QOpenGLShaderProgram,
                            QOpenGLVertexArrayObject, QOpenGLBuffer,
                            QOpenGLContext, QOpenGLVersionProfile,
                            QOpenGLTexture, QImage)
    opengl_versions = ((4, 5), (4, 4), (4, 3), (4, 2), (4, 1), (4, 0), (3, 3))


except ImportError:
    from PyQt6.QtCore import qCritical, QT_VERSION_STR
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from PyQt6.QtOpenGL import (QOpenGLShader, QOpenGLShaderProgram,
                         QOpenGLVertexArrayObject, QOpenGLBuffer,
                         QOpenGLVersionProfile, QOpenGLTexture, QOpenGLVersionFunctionsFactory)
    from PyQt6.QtGui import QSurfaceFormat, QOpenGLContext, QImage
    
    opengl_versions = ((4, 1), (2, 1), (2, 0))


try:
    from collections.abc import Collection, MutableMapping
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection, MutableMapping
import ctypes
import numpy as np



import bluesky as bs
from bluesky.core import Entity
from bluesky.stack import command
from bluesky.ui.qtgl.dds import DDSTexture


# Register settings defaults
bs.settings.set_variable_defaults(gfx_path='graphics')

GLVariable = namedtuple('GLVariable', ['loc', 'size'])
# Convenience object with all GL functions
gl = None
_glvar_sizes = dict()


def get_profile_settings():
    for version in opengl_versions:
        for profile in ('Core', 'Compatibility'):
            try:
                if QT_VERSION_STR[0] == '5':
                    importlib.import_module(f'PyQt5._QOpenGLFunctions_{version[0]}_{version[1]}_{profile}')
                elif QT_VERSION_STR[0] == '6':
                    importlib.import_module(f'PyQt6.QtOpenGL', package=f'QOpenGLFunctions_{version[0]}_{version[1]}_{profile}')
                
                print(f'Found Qt-provided OpenGL functions for OpenGL {version} {profile}')
                return version, QSurfaceFormat.OpenGLContextProfile.CoreProfile if profile == 'Core' else QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile
            except:
                continue
    return (4, 1), None


def init():
    ''' Startup initialisation of OpenGL. '''
    if gl is None:
        version, profile = get_profile_settings()
        # Initialise application-wide GL version
        fmt = QSurfaceFormat()
        fmt.setVersion(*version)
        # profile = QSurfaceFormat.CoreProfile if sys.platform == 'darwin' else QSurfaceFormat.CompatibilityProfile

        fmt.setProfile(profile or QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        QSurfaceFormat.setDefaultFormat(fmt)

        if profile is not None:
            # Use a dummy context to get GL functions
            glprofile = QOpenGLVersionProfile(fmt)
            ctx = QOpenGLContext()
            if QT_VERSION_STR[0] == '5':
                globals()['gl'] = ctx.versionFunctions(glprofile)
            elif QT_VERSION_STR[0] == '6':
                function_factory = QOpenGLVersionFunctionsFactory()
                globals()['gl'] = function_factory.get(glprofile, ctx)
            # Check and set OpenGL capabilities
            if not glprofile.hasProfiles():
                raise RuntimeError(
                    'No OpenGL version >= 3.3 support detected for this system!')
        else:
            # If profile was none, PyQt6 is not shipped with any OpenGL function modules. Use PyOpenGL instead
            print("Couldn't find OpenGL functions in Qt. Falling back to PyOpenGL")
            globals()['gl'] = importlib.import_module('OpenGL.GL')

        globals()['_glvar_sizes'] = {
            gl.GL_FLOAT: 1, gl.GL_FLOAT_VEC2: 2, gl.GL_FLOAT_VEC3: 3,
            gl.GL_FLOAT_VEC4: 4, gl.GL_FLOAT_MAT2: 4, gl.GL_FLOAT_MAT3: 9,
            gl.GL_FLOAT_MAT4: 16, gl.GL_FLOAT_MAT2x3: 6, gl.GL_FLOAT_MAT2x4: 8,
            gl.GL_FLOAT_MAT3x2: 6, gl.GL_FLOAT_MAT3x4: 12, gl.GL_FLOAT_MAT4x2: 8,
            gl.GL_FLOAT_MAT4x3: 12, gl.GL_INT: 1, gl.GL_INT_VEC2: 2, gl.GL_INT_VEC3: 3,
            gl.GL_INT_VEC4: 4, gl.GL_UNSIGNED_INT: 1, gl.GL_UNSIGNED_INT_VEC2: 2,
            gl.GL_UNSIGNED_INT_VEC3: 3, gl.GL_UNSIGNED_INT_VEC4: 4, gl.GL_DOUBLE: 1,
            gl.GL_DOUBLE_VEC2: 2, gl.GL_DOUBLE_VEC3: 3, gl.GL_DOUBLE_VEC4: 4,
            gl.GL_DOUBLE_MAT2: 4, gl.GL_DOUBLE_MAT3: 9, gl.GL_DOUBLE_MAT4: 16,
            gl.GL_DOUBLE_MAT2x3: 6, gl.GL_DOUBLE_MAT2x4: 8, gl.GL_DOUBLE_MAT3x2: 6,
            gl.GL_DOUBLE_MAT3x4: 12, gl.GL_DOUBLE_MAT4x2: 8, gl.GL_DOUBLE_MAT4x3: 12}

    return gl


if gl is None:
    init()


def init_glcontext(ctx):
    ''' Correct OpenGL functions can only be obtained from a valid GL context. '''
    if getattr(gl, '__name__', '') != 'OpenGL.GL':
        # The OpenGL functions are provided by the Qt library. Update them from the current context
        fmt = QSurfaceFormat.defaultFormat()
        glprofile = QOpenGLVersionProfile(fmt)
        if QT_VERSION_STR[0] == '5':
            globals()['gl'] = ctx.versionFunctions(glprofile)
        elif QT_VERSION_STR[0] == '6':
            function_factory = QOpenGLVersionFunctionsFactory()
            globals()['gl'] = function_factory.get(glprofile, ctx)
    # QtOpenGL doesn't wrap all necessary functions. We can do this manually

    # void glGetActiveUniformBlockName(	GLuint program,
    #                                   GLuint uniformBlockIndex,
    #                                   GLsizei bufSize,
    #                                   GLsizei * length,
    #                                   GLchar * uniformBlockName)

    funtype = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_char * 20))
    funcptr = ctx.getProcAddress(b'glGetActiveUniformBlockName')
    c_getuboname = funtype(funcptr.__int__())

    def p_getuboname(programid, uboid):
        name = (ctypes.c_char * 20)()
        c_getuboname(programid, uboid, 20, None, ctypes.pointer(name))
        return name.value.decode('utf-8')

    gl.glGetActiveUniformBlockName = p_getuboname

    # void glGetActiveUniformBlockiv( GLuint program,
    #                                 GLuint uniformBlockIndex,
    #                                 GLenum pname,
    #                                 GLint * params)
    funtype = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.POINTER(ctypes.c_int32))
    funcptr = ctx.getProcAddress(b'glGetActiveUniformBlockiv')
    c_getuboiv = funtype(funcptr.__int__())

    def p_getuboiv(programid, uboid, pname):
        if pname == gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES:
            # Special case: array with specific size is expected
            n_indices = ctypes.c_int32()
            c_getuboiv(programid, uboid,
                       gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, ctypes.pointer(n_indices))
            indices = (ctypes.c_int32 * n_indices.value)()
            # Apparently we need to specifically wrap depending on array size
            funtype = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                       ctypes.POINTER(ctypes.c_int32 * n_indices.value))
            funcptr = ctx.getProcAddress(b'glGetActiveUniformBlockiv')
            c_getuboindices = funtype(funcptr.__int__())
            c_getuboindices(programid, uboid, pname, ctypes.pointer(indices))
            return list(indices)

        param = ctypes.c_int32()
        c_getuboiv(programid, uboid, pname, ctypes.pointer(param))
        return param.value

    gl.glGetActiveUniformBlockiv = p_getuboiv

    # void glVertexAttrib4Nub( GLuint index,
    #                          GLubyte v0,
    #                          GLubyte v1,
    #                          GLubyte v2,
    #                          GLubyte v3)
    funtype = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint8,
                               ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8)
    funcptr = ctx.getProcAddress(b'glVertexAttrib4Nub')
    gl.glVertexAttrib4Nub = funtype(funcptr.__int__())

    # void glTexImage2D( GLenum target,
    #                    GLint level,
    #                    GLint internalFormat,
    #                    GLsizei width,
    #                    GLsizei height,
    #                    GLint border,
    #                    GLenum format,
    #                    GLenum type,
    #                    const void * data)
    funtype = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    funcptr = ctx.getProcAddress(b'glTexImage2D')
    gl.glTexImage2D_alt = funtype(funcptr.__int__())

    # void glTexSubImage2D(	GLenum target,
    #                       GLint level,
    #                       GLint xoffset,
    #                       GLint yoffset,
    #                       GLsizei width,
    #                       GLsizei height,
    #                       GLenum format,
    #                       GLenum type,
    #                       const void * pixels)
    funtype = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32,
                               ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    funcptr = ctx.getProcAddress(b'glTexSubImage2D')
    gl.glTexSubImage2D_alt = funtype(funcptr.__int__())
    
    if getattr(gl, '__name__', '') == 'OpenGL.GL':
        # In case we are using PyOpenGL, get some of the functions to behave in the
        # same way as their counterparts in Qt
        glGetProgramiv_wrap = gl.glGetProgramiv
        def glGetProgramiv(program, pname):
            params = ctypes.c_int32()
            glGetProgramiv_wrap(program, pname, params)
            return params.value
        gl.glGetProgramiv = glGetProgramiv

        glGetActiveAttrib_wrap = gl.glGetActiveAttrib
        def glGetActiveAttrib(program, index):
            length = ctypes.c_int32()
            size = ctypes.c_int32()
            atype = ctypes.c_uint32()
            name = (ctypes.c_char * 20)()
            glGetActiveAttrib_wrap(program, index, 20, length, size, atype, ctypes.pointer(name))
            return name.value.decode('utf-8'), size.value, atype.value
        gl.glGetActiveAttrib = glGetActiveAttrib

        glGetActiveUniform_wrap = gl.glGetActiveUniform
        def glGetActiveUniform(program, index):
            length = ctypes.c_int32()
            size = ctypes.c_int32()
            atype = ctypes.c_uint32()
            name = (ctypes.c_char * 20)()
            glGetActiveUniform_wrap(program, index, 20, length, size, atype, ctypes.pointer(name))
            return name.value.decode('utf-8'), size.value, atype.value
        gl.glGetActiveUniform = glGetActiveUniform

        glTexSubImage3D_wrap = gl.glTexSubImage3D
        def glTexSubImage3D(target, level, xOffset, yOffset, zOffset, width, height, depth, sourceFormat, sourceType, data):
            glTexSubImage3D_wrap(target, level, xOffset, yOffset, zOffset, width,
                                height, depth, sourceFormat, sourceType, ctypes.c_void_p(int(data)))
        gl.glTexSubImage3D = glTexSubImage3D

    
        # The extended initialisation of the remaining GL functions in this function
        # is only required for Qt-provided GL functions
        return

    # GLuint glGetUniformBlockIndex( GLuint program,
    #                                const GLchar * uniformBlockName)
    funtype = ctypes.CFUNCTYPE(
        ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char))
    funcptr = ctx.getProcAddress(b'glGetUniformBlockIndex')
    c_getuboindex = funtype(funcptr.__int__())

    def p_getuboindex(programid, uboname):
        ret = c_getuboindex(programid, ctypes.c_char_p(
            uboname.encode('utf-8')))
        return ret

    gl.glGetUniformBlockIndex = p_getuboindex

    # void glVertexAttribIPointer(GLuint index,
    #                             GLint size,
    #                             GLenum type,
    #                             GLsizei stride,
    #                             const void * pointer)


    funtype = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32, 
                               ctypes.c_uint32, ctypes.c_void_p)
    funcptr = ctx.getProcAddress(b'glVertexAttribIPointer')
    # c_getvapointer = funtype(funcptr.__int__())
    # def p_getvapointer(index, size, atype, normalized, stride, offset):
    #     norm = ctypes.c_bool(0)
    #     c_getvapointer(index, size, atype, norm, stride, offset)
    gl.glVertexAttribIPointer = funtype(funcptr.__int__())


class ShaderSet(MutableMapping):
    ''' A set of shader programs for BlueSky.

        Convenience class to easily switch between sets of shader programs
        (e.g., between the radarwidget and the nd.)

        Normally, each set contains at least the following programs:
        'normal':   Rendering of regular, untextured shapes
        'textured': Rendering of regular, textured shapes
        'text':     Rendering of text objects
        'ssd':      Rendering of SSD objects
    '''
    # Currently selected shader set
    selected = None

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self._programs = dict()
        self._ubos = dict()
        self._spath = ''
        self._iscreated = False
        if ShaderSet.selected is None:
            self.select()

    def create(self):
        ''' Overload this function for creation of shader programs in this set. '''
        self._iscreated = True

    def created(self):
        ''' Returns True if this shaderset was successfully created. '''
        return self._iscreated

    def select(self):
        ''' Select this shader set. '''
        ShaderSet.selected = self

    def update_ubo(self, uboname, *args, **kwargs):
        ''' Update an uniform buffer object of this shader set. '''
        ubo = self._ubos.get(uboname, None)
        if not ubo:
            raise KeyError('Uniform Buffer Object', uboname,
                           'not found in shader set.')
        ubo.update(*args, **kwargs)

    def set_shader_path(self, shader_path):
        ''' Set a search path for shader files. '''
        self._spath = shader_path

    def load_shader(self, shader_name, vs, fs, gs=None):
        ''' Load a shader into this shader set.
            default shader names are: normal, textured, and text. '''
        vs = path.join(self._spath, vs)
        fs = path.join(self._spath, fs)
        if gs:
            gs = path.join(self._spath, gs)

        newshader = ShaderProgram(self.parent)
        if newshader.create(vs, fs, gs):
            self[shader_name] = newshader

    def release_all(self):
        ''' Release bound shader from context. '''
        if ShaderProgram.bound_shader:
            ShaderProgram.bound_shader.release()
            ShaderProgram.bound_shader = None

    def __getitem__(self, key):
        ret = self._programs.get(key, None)
        if not ret:
            raise KeyError('Shader program', key, 'not found in shader set.')
        return ret

    def __setitem__(self, key, program):
        if not isinstance(program, ShaderProgram):
            raise ValueError(
                'Only ShaderProgram objects can be added to a ShaderSet')
        self._programs[key] = program
        # Bind UBO buffers of this shader set to program's UBO's
        for name, size in program.ubos.items():
            ubo = self._ubos.get(name, None)
            if ubo is None:
                ubo = UniformBufferObject()
                ubo.create(size)
                self._ubos[name] = ubo

            program.bind_uniform_buffer(name, ubo)

    def __delitem__(self, key):
        del self._programs[key]

    def __iter__(self):
        return iter(self._programs)

    def __len__(self):
        return len(self._programs)

    @classmethod
    def get_shader(cls, shader_type=''):
        ''' Get a shader from the current shaderset by type. '''
        if not shader_type:
            return ShaderProgram.bound_shader
        ret = cls.selected._programs.get(shader_type, None)
        if not ret:
            raise KeyError('Shader program', shader_type,
                           'not found in shader set.')
        return ret


class ShaderProgram(QOpenGLShaderProgram):
    ''' BlueSky wrapper class for OpenGL shader programs. '''
    bound_shader = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.attribs = dict()
        self.ubos = dict()
        self.uniforms = dict()

    def bind(self):
        ''' Bind this shader to the current context. '''
        if ShaderProgram.bound_shader is not self:
            ShaderProgram.bound_shader = self
            super().bind()

    def release(self):
        ''' Release this shader from the current context. '''
        if ShaderProgram.bound_shader is self:
            super().release()
            ShaderProgram.bound_shader = None

    def create(self, fname_vertex, fname_frag, fname_geom=''):
        ''' Compile shaders and link program.
            Typically executed in initializeGL(). '''
        # Compile shaders from file
        success = True and \
            self.addShaderFromSourceFile(QOpenGLShader.ShaderTypeBit.Vertex, fname_vertex) and \
            self.addShaderFromSourceFile(QOpenGLShader.ShaderTypeBit.Fragment, fname_frag)
        if fname_geom:
            success = success and \
                self.addShaderFromSourceFile(QOpenGLShader.ShaderTypeBit.Geometry, fname_geom)

        # Link program
        if success:
            success = success and self.link()

        if not success:
            print('Shader program creation unsuccessful:')
            print(self.log())
            return False

        # Obtain list of attributes with location and size info
        n_attrs = gl.glGetProgramiv(self.programId(), gl.GL_ACTIVE_ATTRIBUTES)
        for a in range(n_attrs):
            name, size, atype = gl.glGetActiveAttrib(self.programId(), a)
            loc = self.attributeLocation(name)
            typesize = _glvar_sizes.get(atype, 1)
            self.attribs[name] = GLVariable(loc, size * typesize)

        # Get number of uniforms
        n_uniforms = gl.glGetProgramiv(self.programId(), gl.GL_ACTIVE_UNIFORMS)
        all_uids = set(range(n_uniforms))

        # Obtain list of uniform blocks
        n_ub = gl.glGetProgramiv(self.programId(), gl.GL_ACTIVE_UNIFORM_BLOCKS)
        for ub in range(n_ub):
            name = gl.glGetActiveUniformBlockName(self.programId(), ub)
            size = gl.glGetActiveUniformBlockiv(
                self.programId(), ub, gl.GL_UNIFORM_BLOCK_DATA_SIZE)
            self.ubos[name] = size
            # ubsize = gl.glGetActiveUniformBlockiv(self.programId(), ub,
            # gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS)
            indices = gl.glGetActiveUniformBlockiv(
                self.programId(), ub, gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES)
            all_uids -= set(indices)
            # print('Uniform block: ', name.value.decode('utf8'),
            #   'size =', ubsize.value)
            # for i in indices:
            #     usize = ctypes.c_int32()
            #     utype = ctypes.c_int32()
            #     uname = (ctpes.c_char * 20)()
            #     gl.glGetActiveUniform(self.programId(), i, 20, None,
            #         ctypes.pointer(usize), ctypes.pointer(utype),
            #         ctypes.pointer(uname))
            #     print('block uniform', i, '=', uname.value.decode('utf8'),
            #           'size =', usize.value)
            #     print(gl.glGetActiveUniform(self.programId(), i))

        # Obtain list of remaining uniforms (those not part of a block)
        for u in all_uids:
            name, size, utype = gl.glGetActiveUniform(self.programId(), u)
            self.uniforms[name] = GLVariable(
                u, size * _glvar_sizes.get(utype, 1))

        return True

    def bind_uniform_buffer(self, ubo_name, ubo):
        ''' Bind a uniform buffer block to this shader. '''
        idx = gl.glGetUniformBlockIndex(self.programId(), ubo_name)
        gl.glUniformBlockBinding(self.programId(), idx, ubo.binding)


class VertexArrayObject(QOpenGLVertexArrayObject):
    ''' Wrapper around the OpenGL approach to drawing a single shape from one
        or more buffers. '''

    def __init__(self, primitive_type=gl.GL_LINE_STRIP, shader_type='normal', parent=None):
        super().__init__(parent)
        self.shader_type = shader_type
        self.texture = None
        self.primitive_type = primitive_type
        self.first_vertex = 0
        self.vertex_count = 0
        self.n_instances = 0
        self.max_instance_divisor = 0
        self.single_color = None
        self.single_scale = None

    def set_primitive_type(self, primitive_type):
        ''' Set the primitive type for this VAO. '''
        self.primitive_type = primitive_type

    def set_vertex_count(self, count):
        ''' Set the vertex count for this VAO. '''
        self.vertex_count = int(count)

    def set_first_vertex(self, vertex):
        ''' Set the first vertex for this VAO. '''
        self.first_vertex = vertex

    def create(self, texture=None, vertex_count=0, n_instances=0, **attribs):
        ''' Create the actual VAO, attach passed attribs, and potentially
            create new buffers. '''
        super().create()
        if texture is not None:
            if isinstance(texture, Texture):
                self.texture = texture
            else:
                self.texture = Texture()
                self.texture.load(texture)
            if self.shader_type == 'normal':
                self.shader_type = 'textured'
        self.vertex_count = vertex_count
        self.n_instances = n_instances

        # Update special attribute 'scale' if necessary
        scaleattrib = ShaderSet.get_shader(self.shader_type).attribs.get('scale')
        if scaleattrib is not None:
            self.single_scale = (scaleattrib.loc, 1.0)

        self.set_attribs(**attribs)

    def set_attribs(self, usage=QOpenGLBuffer.UsagePattern.StaticDraw, instance_divisor=0,
                    datatype=None, stride=0, offset=None, normalize=False,
                    **attribs):
        ''' Set attributes for this VAO. '''
        if not attribs:
            return
        self.bind()
        program = ShaderSet.get_shader(self.shader_type)
        for name, data in attribs.items():
            attrib = program.attribs.get(name, None)
            if not attrib:
                raise KeyError('Unknown attribute ' + name +
                               ' for shader type ' + self.shader_type)

            dtype = datatype or (gl.GL_UNSIGNED_BYTE if name == 'color' else gl.GL_FLOAT)
            if name == 'color':
                normalize = True

            if isinstance(data, QOpenGLBuffer):
                # A previously created GL buffer is passed
                buf = data
                buf.bind()
            elif isinstance(data, Collection):
                # Color attribute has a special condition for a single color
                if name == 'color' and np.size(data) in (3, 4):
                    # Store attrib location and color
                    self.single_color = np.append(attrib.loc, data)
                    if len(data) == 3:
                        # Add full alpha if none is given
                        self.single_color = np.append(self.single_color, 255)
                    continue
                # If the input is an array create a new GL buffer
                buf = GLBuffer()
                buf.create(usage=usage, data=data)
            elif isinstance(data, int):
                buf = GLBuffer()
                buf.create(usage=usage, size=data)

            # Special attribs: scale and vertex
            if name == 'vertex' and isinstance(data, Collection):
                self.vertex_count = np.size(data) // 2
            elif name == 'scale':
                if isinstance(data, float):
                    self.single_scale = (attrib.loc, data)
                    continue
                else:
                    self.single_scale = None

            # Bind the buffer to the indicated attribute for this VAO
            program.enableAttributeArray(attrib.loc)
            if dtype == gl.GL_UNSIGNED_BYTE and not normalize:
                gl.glVertexAttribIPointer(attrib.loc, attrib.size, dtype, stride, offset)
            else:
                program.setAttributeBuffer(attrib.loc, dtype, 0, attrib.size)

            # For instanced data, indicate per how many instances
            # we move a step in the buffer (1=per instance)
            if instance_divisor > 0:
                gl.glVertexAttribDivisor(attrib.loc, instance_divisor)
                self.max_instance_divisor = max(
                    instance_divisor, self.max_instance_divisor)

            # Keep a reference to the buffer of this attribute
            setattr(self, name, buf)

    def update(self, **attribs):
        ''' Update one or more buffers for this object. '''
        for name, data in attribs.items():
            attrib = getattr(self, name, None)
            if not isinstance(attrib, GLBuffer):
                raise KeyError('Unknown attribute ' + name)
            # Special attribs: color and vertex
            if name == 'vertex' and isinstance(data, Collection):
                self.vertex_count = np.size(data) // 2

            # Update the buffer of the attribute
            attrib.update(data)

    def draw(self, primitive_type=None, first_vertex=None, vertex_count=None, n_instances=None):
        ''' Draw this VAO. '''
        if primitive_type is None:
            primitive_type = self.primitive_type

        if first_vertex is None:
            first_vertex = self.first_vertex

        if vertex_count is None:
            vertex_count = self.vertex_count

        if n_instances is None:
            n_instances = self.n_instances

        if vertex_count == 0:
            return
        shader = ShaderSet.get_shader(self.shader_type)
        shader.bind()
        self.bind()

        if self.single_color is not None:
            gl.glVertexAttrib4Nub(*self.single_color)
        elif self.texture:
            self.texture.bind(0)

        if self.single_scale is not None:
            shader.setAttributeValue(*self.single_scale)

        if n_instances > 0:
            gl.glDrawArraysInstanced(
                primitive_type, first_vertex, vertex_count, n_instances * self.max_instance_divisor)
        else:
            gl.glDrawArrays(primitive_type, first_vertex, vertex_count)


class Text(VertexArrayObject):
    ''' Convenience class for a text object. '''

    def __init__(self, charsize=16.0, blocksize=None, font=None):
        super().__init__(primitive_type=gl.GL_TRIANGLES, shader_type='text')
        self.font = font or Font.getdefault()
        self.blocksize = blocksize or (0, 0)
        self.charsize = charsize

    def create(self, text, lat=None, lon=None, color=None, vertex_offset=None, instanced=False):
        ''' Create the Text VAO. '''
        super().create()
        if not self.font.isCreated():
            self.font.create()
        w, h = self.charsize, self.charsize * self.font.char_ar
        x, y = vertex_offset or (0.0, 0.0)
        if instanced:
            vertices, texcoords = self.font.char(x, y, w, h)
        else:
            vertices, texcoords = [], []
            for i, c in enumerate(text):
                v, t = self.font.char(x + i * w, y, w, h, ord(c))
                vertices += v
                texcoords += t

        self.set_attribs(vertex=np.array(vertices, dtype=np.float32),
                         texcoords=np.array(texcoords, dtype=np.float32))

        if instanced:
            self.set_attribs(texdepth=text, instance_divisor=1,
                             datatype=gl.GL_UNSIGNED_BYTE)
        divisor = self.blocksize[0] * self.blocksize[1] if instanced else 0
        if lat is not None and lon is not None:
            self.set_attribs(lat=lat, lon=lon, instance_divisor=divisor)

        if color:
            self.set_attribs(color=color, instance_divisor=divisor)

    def draw(self, first_vertex=None, vertex_count=None, n_instances=None):
        ''' Draw this text VAO. '''
        ShaderSet.get_shader(self.shader_type).bind()
        self.font.bind()
        self.font.set_char_size(self.charsize)
        if self.blocksize:
            self.font.set_block_size(self.blocksize)
        super().draw(first_vertex=first_vertex,
                     vertex_count=vertex_count, n_instances=n_instances)


class RenderTarget:
    ''' Wrapper class for OpenGL render targets, which can be widgets inside a bigger window,
        or independent GL windows. '''
    __rendertargets__ = dict()

    def __init__(self, shaderset=None):
        # Store all RenderTarget objects
        RenderTarget.__rendertargets__[self.__class__.__name__.upper()] = self
        self._renderobjs = OrderedDict()
        self._shaderset = shaderset

    def set_shaderset(self, shset):
        ''' Set the shaderset for this render target. '''
        self._shaderset = shset

    def addobject(self, obj):
        ''' Add a RenderObject to this target to draw.
            Objects are stored and drawn sorted by layer. '''
        self._renderobjs[obj.getbase().name()] = obj
        self._renderobjs = OrderedDict(sorted(self._renderobjs.items(),
                                             key=lambda o: o[1].layer))


class RenderWidget(QOpenGLWidget, RenderTarget):
    def initializeGL(self):
        init_glcontext(self.context())
        # First check for supported GL version
        gl_version = float(gl.glGetString(gl.GL_VERSION)[:3])
        if gl_version < 3.3:
            print(('OpenGL context created with GL version %.1f' % gl_version))
            qCritical("""Your system reports that it supports OpenGL up to version %.1f. The minimum requirement for BlueSky is OpenGL 3.3.
                Generally, AMD/ATI/nVidia cards from 2008 and newer support OpenGL 3.3, and Intel integrated graphics from the Haswell
                generation and newer. If you think your graphics system should be able to support GL>=3.3 please open an issue report
                on the BlueSky Github page (https://github.com/TUDelft-CNS-ATM/bluesky/issues)""" % gl_version)
            return

        if self._shaderset is None and self._renderobjs:
            qCritical("Cannot create render objects without an initialised shader set!")
            return

        # Initialise our shaderset if this hasn't been done yet
        if not self._shaderset.created():
            self._shaderset.create()

        # Call the create method of all registered objects
        for obj in self._renderobjs.values():
            obj.create()

    def paintGL(self):
        """Paint the scene."""
        # clear the framebuffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Draw our render objects in order of layer
        for obj in self._renderobjs.values():
            obj.draw()

        # Release shaders
        self._shaderset.release_all()


class RenderObject(Entity, skipbase=True):
    ''' Convenience singleton class for drawing different (nested) objects. '''
    # Known RenderObject base classes
    __renderobjs__ = dict()

    def __init_subclass__(cls, layer=0, skipbase=False):
        # All renderobjects are replaceable, but it is still possible to create an intermediate non-instantiable base class
        if not skipbase and not hasattr(cls, '_baseimpl'):
            # Store passed layer as a class variable
            cls.layer = layer
            cls.visible = True
            # Store all RenderObject base implementations
            RenderObject.__renderobjs__[cls.__name__.upper()] = cls

        return super().__init_subclass__(replaceable=True, skipbase=skipbase)

    def __init__(self, parent=None):
        self.parent = parent or self.getdefault().implinstance().parent
        self.glsurface = self.parent.glsurface if isinstance(self.parent, RenderObject) else self.parent
        self.children = list()

    def draw(self):
        ''' Draw this object. '''
        for child in self.children:
            child.draw()

    def create(self):
        ''' Create this object. '''
        pass

    @property
    def shaderset(self):
        ''' The shaderset of a RenderObject.
            Always points to the currently selected ShaderSet.
        '''
        return ShaderSet.selected


@command(aliases=('ADDVIS',))
def addvisual(objname: "txt" = "", target: "txt" = "RADARWIDGET"):
    ''' Add a render object to a render target. 
    
        Argements:
        - obj: The renderobject to add. 
        - target: A render target such as the RadarWidget (the default) and the ND.
    '''
    if not target:
        return True, f'Available render targets: {", ".join(RenderTarget.__rendertargets__)}'
    
    targetobj = RenderTarget.__rendertargets__.get(target)
    if not targetobj:
        return False, f'Render target {target} not found!\n' + \
            f'Available render targets: {", ".join(RenderTarget.__rendertargets__)}'

    if not objname:
        existing = targetobj._renderobjs
        canadd = set(RenderObject.__renderobjs__.keys()) - set(existing)
        msg = f'Target {target} is currently drawing the following objects:\n'
        msg += ', '.join(existing) + '\n'
        if canadd:
            msg += f'Further objects that can be added to {target} are:\n'
            msg += ', '.join(canadd)
        else:
            msg += 'There are no further objects available to add.'
        return True, msg

    classobj = RenderObject.__renderobjs__.get(objname)
    if not classobj:
        return False, f'Unknown render object: {objname}!'
    # Check if object is already instantiated
    firsttime = not classobj.is_instantiated()
    obj = classobj(parent=targetobj)
    if firsttime:
        targetobj.makeCurrent()
        obj.create()
    targetobj.addobject(obj)


@command(aliases=('VIS',))
def visual(objname: "txt" = "", vis: "bool/txt" = ""):
    ''' Set the appearance and visibility of render objects. '''
    if not objname:
        return True, "Render objects in BlueSky:\n" + ", ".join(RenderObject.__renderobjs__)
    baseimpl = RenderObject.__renderobjs__.get(objname)
    if not baseimpl:
        return False, f"Render object {objname} not known!"
    # If vis is a boolean, this command is meant to toggle the visibility of the render object
    if isinstance(vis, bool):
        baseimpl.visible = vis
        return True, f'setting visibility for {objname} to {vis}'
    all_impls = baseimpl.derived()
    if vis == "":
        return True, f"{objname} has the following available implementations:\n" + ", ".join(all_impls)

    impl = all_impls.get(vis)
    if impl is None:
        return False, f'{vis} doesn\'t exist.\n' + \
            f"{objname} has the following available implementations:\n" + \
            ", ".join(all_impls)
    # Implementation exists, we can select it
    # Check first if implementation was selected before, otherwise create should be called
    # after construction
    firsttime = not impl.is_instantiated()
    impl.select()
    if firsttime:
        impl.instance().glsurface.makeCurrent()
        impl.instance().create()
    return True, f'Selected {vis} as visualisation for {objname}.'


class Circle(VertexArrayObject):
    ''' Convenience class for a circle. '''

    def __init__(self, shader_type='normal', parent=None):
        super().__init__(gl.GL_LINE_LOOP, shader_type, parent)

    def create(self, radius, nsegments=36, **attribs):
        ''' Create the Circle VAO.'''
        vertices = [(radius * np.cos(i / nsegments * 2.0 * np.pi),
                     radius * np.sin(i / nsegments * 2.0 * np.pi)) for i in range(nsegments)]
        # vcircle = np.array(np.transpose((
        #     radius * np.cos(np.linspace(0.0, 2.0 * np.pi, nsegments)),
        #     radius * np.sin(np.linspace(0.0, 2.0 * np.pi, nsegments)))), dtype=np.float32)
        super().create(vertex=np.array(vertices, dtype=np.float32), **attribs)

class Rectangle(VertexArrayObject):
    ''' Convenience class for a rectangle. '''

    def __init__(self, shader_type='normal', parent=None):
        super().__init__(gl.GL_LINE_LOOP, shader_type, parent)

    def create(self, w, h, fill=False):
        ''' Create the Rectangle VAO.'''
        if fill:
            self.set_primitive_type(gl.GL_TRIANGLE_FAN)
        vrect = np.array([(-0.5 * h, 0.5 * w), (-0.5 * h, -0.5 * w),
                          (0.5 * h, -0.5 * w), (0.5 * h, 0.5 * w)], dtype=np.float32)
        super().create(vertex=vrect)


class GLBuffer(QOpenGLBuffer):
    ''' Wrapper class for vertex and index buffers. '''

    def create(self, size=None, usage=QOpenGLBuffer.UsagePattern.StaticDraw, data=None):
        ''' Create the buffer. '''
        if size is None and data is None:
            raise ValueError(
                'Either a size or a set of data should be provided when creating a GL buffer')
        super().create()
        self.setUsagePattern(usage)
        self.bind()
        if data is not None:
            bufdata, size = content_and_size(data)
            self.allocate(bufdata, size)
        else:
            self.allocate(size)
        # Test if allocated size is as requested
        if self.size() != size:
            print(f'GLBuffer: Warning: could not allocate buffer of size {size}. Actual size is {self.size()}')

    def update(self, data, offset=0, size=None):
        ''' Send new data to this GL buffer. '''
        dbuf, dsize = content_and_size(data)
        size = size or dsize
        self.bind()
        if size > self.size() - offset:
            print(f'GLBuffer: Warning, trying to send more data ({size} bytes)'
                  f'to buffer than allocated size ({self.size()} bytes).')
            size = self.size() - offset
        
        self.write(offset, dbuf, size)
        # TODO: master branch has try/except for buffer writes after closing context


class UniformBufferObject(GLBuffer):
    ''' Wrapper class for uniform buffers. '''
    ufo_max_binding = 1

    def __init__(self):
        super().__init__(QOpenGLBuffer.Type.VertexBuffer)
        self.binding = 0

    def create(self, size=None, usage=QOpenGLBuffer.UsagePattern.StaticDraw, data=None):
        ''' Create this UBO. '''
        super().create(size, usage, data)
        self.binding = UniformBufferObject.ufo_max_binding
        UniformBufferObject.ufo_max_binding += 1
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER,
                            self.binding, self.bufferId())


class Texture(QOpenGLTexture):
    ''' BlueSky OpenGL Texture class. '''
    def __init__(self, target=QOpenGLTexture.Target.Target2D):
        super().__init__(target)

    def load(self, fname):
        ''' Load the texture into GPU memory. '''
        fname = Path(fname)
        if fname.suffix.lower() == '.dds':
            tex = DDSTexture(fname)
            self.setFormat(QOpenGLTexture.TextureFormat.RGB_DXT1)
            self.setSize(tex.width, tex.height)
            self.setWrapMode(QOpenGLTexture.WrapMode.Repeat)
            self.allocateStorage()
            self.setCompressedData(len(tex.data), tex.data)
        else:
            self.setData(QImage(fname.as_posix()))
            self.setWrapMode(QOpenGLTexture.WrapMode.Repeat)

    def bind(self, unit=0):
        # Overload texture bind with default texture unit to make sure that, unless an explicit
        # texture unit is passed, the default is always zero.
        super().bind(unit)

    def setLayerData(self, layer, image):
        ''' For array textures, set the image data at given layer. '''
        ptr = image.constBits()
        ptr.setsize(image.width() * image.height() * 4)

        # xOffset, yOffset, zOffset, width, height, depth, sourceFormat, sourceType, data
        # self.setData(0, 0, i - 30, imgsize[0], imgsize[1], 1,
        #              QOpenGLTexture.BGRA, QOpenGLTexture.UInt8, ptr) #Qt >= 5.14
        gl.glTexSubImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer,
                           image.width(), image.height(), 1, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, ptr)

class Font(Texture):
    ''' BlueSky class to implement a font using a GL Texture array. '''
    _fonts = list()

    def __init__(self):
        super().__init__(QOpenGLTexture.Target.Target2DArray)
        self.char_ar = 1.0
        self.loc_char_size = 0
        self.loc_block_size = 0
        Font._fonts.append(self)

    def create(self):
        ''' Create this font. '''
        txtshader = ShaderSet.get_shader('text')
        self.loc_char_size = txtshader.uniforms['char_size'].loc
        self.loc_block_size = txtshader.uniforms['block_size'].loc

        # Load the first image to get font size
        fname = (bs.resource(bs.settings.gfx_path) / 'font/{i}.png').as_posix()
        img = QImage(fname.format(i=32))
        imgsize = (img.width(), img.height())
        self.char_ar = float(imgsize[1]) / imgsize[0]

        super().create()
        self.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        self.setSize(img.width(), img.height())
        self.setLayers(127 - 30)

        self.bind()
        self.allocateStorage()
        # gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, gl.GL_RGBA8,
        #                 imgsize[0], imgsize[1], 127 - 30, 0, gl.GL_BGRA,
        #                 gl.GL_UNSIGNED_BYTE, None)
        self.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionS,
                         QOpenGLTexture.WrapMode.ClampToBorder)
        self.setWrapMode(QOpenGLTexture.CoordinateDirection.DirectionT,
                         QOpenGLTexture.WrapMode.ClampToBorder)
        self.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)

        # We're using the ASCII range 32-126; space, uppercase, lower case,
        # numbers, brackets, punctuation marks
        for i in range(30, 127):
            img = QImage(fname.format(i=i)).convertToFormat(QImage.Format.Format_ARGB32)
            self.setLayerData(i - 30, img)

    @classmethod
    def getdefault(cls):
        ''' Get the default font. '''
        if not cls._fonts:
            return Font()
        return cls._fonts[0]

    def set_char_size(self, char_size):
        ''' Set the character size uniform. '''
        gl.glUniform2f(self.loc_char_size, char_size, char_size * self.char_ar)

    def set_block_size(self, block_size):
        ''' Set the block size uniform. '''
        gl.glUniform2i(self.loc_block_size, block_size[0], block_size[1])

    @staticmethod
    def char(x, y, w, h, c=32):
        ''' Convenience function to get vertices and texture coordinates for a
            single character. '''
        # Two triangles per character
        vertices = [(x, y + h), (x, y), (x + w, y + h),
                    (x + w, y + h), (x, y), (x + w, y)]
        texcoords = [(0, 0, c), (0, 1, c), (1, 0, c),
                     (1, 0, c), (0, 1, c), (1, 1, c)]
        return vertices, texcoords


def content_and_size(data):
    ''' Convenience function to get the correct variables to upload data to
        GL buffers. '''
    if isinstance(data, np.ndarray):
        return data, data.nbytes
    if isinstance(data, (ctypes.Structure, ctypes.Array)):
        # return ctypes.pointer(data), ctypes.sizeof(data)
        return bytes(data), ctypes.sizeof(data)
    return None, 0
