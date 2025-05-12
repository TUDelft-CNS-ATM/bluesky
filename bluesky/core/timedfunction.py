import inspect
import functools
from types import SimpleNamespace
from collections import OrderedDict

from bluesky.core.funcobject import FuncObject
from bluesky.core.simtime import Timer


class _Hook(OrderedDict):
    def trigger(self):
        for callback in self.values():
            callback()


# Dictionaries of timed functions for different trigger points
hooks = SimpleNamespace(
    update=_Hook(),
    preupdate=_Hook(),
    hold=_Hook(),
    reset=_Hook()
)


def timed_function(func=None, name='', dt=0, hook='update'):
    ''' Decorator to turn a function into a (periodically) timed function. '''
    def deco(func):
        # Generate a name if none is provided
        if not name:
            if inspect.ismethod(func):
                if inspect.isclass(func.__self__):
                    # classmethods
                    tname = f'{func.__self__.__name__}.{func.__name__}'
                else:
                    # instance methods
                    tname = f'{func.__self__.__class__.__name__}.{func.__name__}'
            else:
                tname = f'{func.__module__}.{func.__name__}'
        else:
            tname = name
        fobj = FuncObject(func)
        if 'update' in hook or 'preupdate' in hook:
            # Generate the wrapped timed callback
            timer = Timer(tname, dt)
            # Create the appropriate call method
            if 'dt' in inspect.signature(func).parameters:
                # Callback for functions that have dt as argument
                @functools.wraps(fobj)
                def callback(*args):
                    if timer.readynext:
                        fobj(*args, dt=float(timer.dt_act))
            else:
                # Callback for functions without dt as argument
                @functools.wraps(fobj)
                def callback(*args):
                    if timer.readynext:
                        fobj(*args)
        else:
            @functools.wraps(fobj)
            def callback(*args):
                fobj(*args)

        # Add automatically-triggered function to appropriate dict(s) if not there yet.
        hooknames = hook if isinstance(hook, (list, tuple)) else (hook,)
        for hookname in hooknames:
            target = getattr(hooks, hookname)
            if target is None:
                raise KeyError(f'No timing hook found with name {hookname}')
            if tname not in target:
                target[tname] = fobj if hookname in ('reset', 'hold') else callback

        # Construct the timed function, but return the original function
        return func

    # Allow both @timed_function and @timed_function(args)
    return deco(func) if func else deco
