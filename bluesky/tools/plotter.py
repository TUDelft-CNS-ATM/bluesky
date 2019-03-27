''' Sim-side implementation of graphical data plotter in BlueSky.'''
from collections import defaultdict
import bluesky as bs
from bluesky.tools import varexplorer as ve


# Globals
# The list of plots
plots = list()


def plot(varx='', vary='', dt=1.0, fig=None, **params):
    ''' Select a set of variables to plot.
        Arguments: varx, vary, dt, color, fig. '''
    try:
        newplot = Plot(varx, vary, dt, fig, **params)
        plots.append(newplot)
        return True
    except IndexError as e:
        return False, e.args[0]


def legend(legend, fig=None):
    ''' Set a legend for a figure. '''
    try:
        # Get the plot with the corresponding figure number
        p = plots[-1] if fig is None else next(
            plot for plot in plots if plot.fig == fig)

        data = {p.fig: dict(legend=legend)}
        bs.net.send_stream(p.stream_id, data)
        return True
    except IndexError as e:
        return False, e.args[0]

def update(simt):
    ''' Periodic update function for the plotter. '''
    streamdata = defaultdict(dict)
    for plot in plots:
        if plot.tnext <= simt:
            plot.tnext += plot.dt
            streamdata[plot.stream_id][plot.fig] = dict(x=plot.x.get(), y=plot.y.get())

    for streamname, data in streamdata.items():
        bs.net.send_stream(streamname, data)


class Plot(object):
    ''' A plot object.
        Each plot object is used to manage the plot of one variable
        on the sim side.'''

    maxfig = 0

    def __init__(self, varx='', vary='', dt=1.0, fig=None, **params):
        self.x = ve.findvar(varx if vary else 'simt')
        self.y = ve.findvar(vary or varx)
        self.dt = dt
        self.tnext = bs.sim.simt
        self.params = params
        if not fig:
            fig = Plot.maxfig
            Plot.maxfig += 1
        elif fig > Plot.maxfig:
            Plot.maxfig = fig

        self.fig = fig

        self.stream_id = b'PLOT' + (bs.stack.sender() or b'*')

        if None in (self.x, self.y):
            raise IndexError('Variable {} not found'.format(varx if self.x is None else (vary or varx)))

        # if not self.x.is_num() or not self.y.is_num():
        #     raise IndexError('Variable {} not numeric'.format(varx if not self.x.is_num() else (vary or varx)))
        bs.net.send_stream(self.stream_id, {self.fig: params})

    def send(self):
        bs.net.send_stream(self.stream_id, {self.fig : dict(x=self.x.get(), y=self.y.get())})