""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
from PyQt6.QtCore import Qt
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, ui  #, settings, navdb, sim, scr, tools
from bluesky.ui.qtgl.glhelpers import gl, RenderObject, VertexArrayObject
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel


class MyWindow(QWidget):
    ''' A simple window with a text label and a button. '''
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.button = QPushButton(text="Click me!")
        self.label = QLabel(text="Hello!")
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        ''' Callback function that is called when our button is clicked. '''
        stack.stack('MCRE 1')


# Create one window
window = MyWindow()


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    ''' Plugin initialisation function. '''
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'QTEXAMPLE',

        # The type of this plugin.
        'plugin_type':     'gui',
        }

    # Show our window when the plugin is loaded
    window.show()

    # init_plugin() should always return a configuration dict.
    return config


@stack.command
def qtexample():
    '''Re-show our window with a stack command. '''
    window.show()