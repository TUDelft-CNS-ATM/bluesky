from ..settings import gui

if gui == 'qtgl':
    from mt import Simulation, MainLoop
elif gui == 'pygame':
    from st import Simulation, MainLoop
