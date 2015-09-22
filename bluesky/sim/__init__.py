from ..settings import gui

if gui == 'qtgl':
    from qtgl import Simulation, MainLoop
elif gui == 'pygame':
    from pygame import Simulation, MainLoop
