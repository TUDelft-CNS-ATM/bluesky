from ..settings import gui

if gui == 'qtgl':
    from qtgl.simulation import Simulation
elif gui == 'pygame':
    from pygame.simulation import Simulation
