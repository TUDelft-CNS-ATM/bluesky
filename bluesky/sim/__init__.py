from ..settings import gui

if gui == 'qtgl':
    from qtgl import Simulation
elif gui == 'pygame':
    from pygame import Simulation
