""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, ui  #, settings, navdb, sim, scr, tools
from bluesky.ui.qtgl.glhelpers import gl, RenderObject, VertexArrayObject


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'GUIEXAMPLE',

        # The type of this plugin.
        'plugin_type':     'gui',
        }

    # init_plugin() should always return a configuration dict.
    return config


### Entities in BlueSky are objects that are created only once (called singleton)
### which implement some traffic or other simulation functionality.
### To define an entity that ADDS functionality to BlueSky, create a class that
### inherits from bluesky.core.Entity.
### To replace existing functionality in BlueSky, inherit from the class that
### provides the original implementation (see for example the asas/eby plugin).
class MyVisual(RenderObject, layer=100):
    ''' Example new render object for BlueSky. '''
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.shape = VertexArrayObject(gl.GL_TRIANGLE_FAN)

    def create(self):
        vertices = np.array([52, 5, 52, 4, 54, 4, 54, 5], dtype = np.float32)
        self.shape.create(vertex=vertices, color=(255, 0, 0))

    def draw(self):
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_LATLON)
        self.shape.draw()

    # You can create new stack commands with the stack.command decorator.
    # By default, the stack command name is set to the function name.
    # The default argument type is a case-sensitive word. You can indicate different
    # types using argument annotations. This is done in the below function:
    # - The color argument returns three integer values (r, g, b), which is why the 
    #   starred notation is used here.
    @stack.command
    def mycolor(self, *color: 'color'):
        ''' Set the color of my example shape. '''
        self.shape.single_color[1:4] = color
