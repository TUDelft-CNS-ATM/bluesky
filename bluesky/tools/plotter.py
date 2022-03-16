''' Sim-side implementation of graphical data plotter in BlueSky.'''
from collections import defaultdict
import bluesky as bs
from bluesky.core import varexplorer as ve


# Globals
# The list of plots
plots = list()


def plot(*args, **params):
    ''' Select a set of variables to plot.
        Arguments: varx, vary, dt, color, fig. '''
    if args:
        try:
            newplot = Plot(*args, **params)
            plots.append(newplot)
        except IndexError as e:
            return False, e.args[0]
    bs.net.send_stream(b'PLOT' + (bs.stack.sender() or b'*'), dict(show=True))
    return True


def legend(legend, fig=None):
    ''' Set a legend for a figure. '''
    try:
        # Get the plot with the corresponding figure number
        p = plots[-1] if fig is None else next(
            plot for plot in plots if plot.fig == str(fig))

        data = {p.fig: dict(legend=legend)}
        bs.net.send_stream(p.stream_id, data)
        return True
    except (IndexError, StopIteration) as e:
        return False, e.args[0]

def reset():
    ''' Remove plots when simulation is reset. '''
    # Notify clients of removal of plots
    notify_ids = {p.stream_id for p in plots}
    for stream_id in notify_ids:
        bs.net.send_stream(stream_id, dict(reset=True))
    plots.clear()


def update():
    ''' Periodic update function for the plotter. '''
    streamdata = defaultdict(dict)
    for p in plots:
        if p.tnext <= bs.sim.simt:
            p.tnext += p.dt
            streamdata[p.stream_id][p.fig] = dict(x=p.x.get(), y=p.y.get())

    for streamname, data in streamdata.items():
        bs.net.send_stream(streamname, data)


class Plot:
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

        self.fig = str(fig)

        self.stream_id = b'PLOT' + (bs.stack.sender() or b'*')

        if None in (self.x, self.y):
            raise IndexError('Variable {} not found'.format(varx if self.x is None else (vary or varx)))

        # if not self.x.is_num() or not self.y.is_num():
        #     raise IndexError('Variable {} not numeric'.format(varx if not self.x.is_num() else (vary or varx)))
        print(self.stream_id, type(self.stream_id), {self.fig: params})
        bs.net.send_stream(self.stream_id, {self.fig: params})

    def send(self):
        bs.net.send_stream(self.stream_id, {self.fig : dict(x=self.x.get(), y=self.y.get())})
