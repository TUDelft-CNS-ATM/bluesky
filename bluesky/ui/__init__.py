from ..settings import gui
if gui == 'qtgl':
    from qtgl import Gui
elif gui == 'pygame':
    from pygame import Gui
