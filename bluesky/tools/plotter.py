''' Sim-side implementation of graphical data plotter in BlueSky.'''
from collections import defaultdict
import bluesky as bs
from bluesky.tools import varexplorer as ve


# Globals
# The list of plots
plots = list()


def plot(*args):
    ''' Stack function to select a set of variables to plot.
        Arguments: varx, vary, dt, color, fig. '''
    try:
        plots.append(Plot(*args))
        return True
    except IndexError as e:
        return False, e.args[0]


def update(simt):
    ''' Periodic update function for the plotter. '''
    streamdata = defaultdict(dict)
    for plot in plots:
        if plot.tnext <= simt:
            plot.tnext += plot.dt
            streamdata[plot.stream_id][plot.fig] = (plot.x.get(), plot.y.get(), plot.color)

    for streamname, data in streamdata.items():
        bs.net.send_stream(streamname, data)


class Plot(object):
    ''' A plot object.
        Each plot object is used to manage the plot of one variable
        on the sim side.'''

    maxfig = 0

    def __init__(self, varx='', vary='', dt=1.0, color=None, fig=None):
        self.x = ve.findvar(varx if vary else 'simt')
        self.y = ve.findvar(vary or varx)
        self.dt = dt
        self.tnext = bs.sim.simt
        self.color = color
        if not fig:
            fig = Plot.maxfig
            Plot.maxfig += 1
        elif fig > Plot.maxfig:
            Plot.maxfig = fig

        self.fig = fig

        self.stream_id = b'PLOT' + bs.stack.sender()

        if None in (self.x, self.y):
            raise IndexError('Variable {} not found'.format(varx if self.x is None else (vary or varx)))

        if not self.x.is_num() or not self.y.is_num():
            raise IndexError('Variable {} not numeric'.format(varx if not self.x.is_num() else (vary or varx)))
