''' Simulation clock with guaranteed decimal precision. '''
import inspect
from collections import OrderedDict
from inspect import signature
from types import SimpleNamespace
from decimal import Decimal
from bluesky import settings


# Register settings defaults
settings.set_variable_defaults(simdt=0.05)

MAX_RECOVERY_FAC = 4

# Data that the simulation clock needs to keep
_clock = SimpleNamespace(t=Decimal('0.0'), dt=Decimal(repr(settings.simdt)),
                         ft=0.0, fdt=settings.simdt)
_timers = OrderedDict()

# Dictionaries of timed functions for different trigger points
preupdate_funs = OrderedDict()
update_funs = OrderedDict()
reset_funs = OrderedDict()


def setdt(newdt=None, target='simdt'):
    ''' Set the timestep for the simulation clock.
        Returns a floating-point representation of the new timestep. '''
    if newdt is None:
        text = f'Simulation timesteps:\nbase dt = {_clock.fdt}'
        for timer in _timers.values():
            text += f'\n{timer.name} = {timer.dt_act}'
        return True, text
    if target == 'simdt':
        _clock.dt = Decimal(repr(newdt))
        _clock.fdt = float(_clock.dt)
        msg = f'Base dt set to {_clock.dt}'
        for timer in _timers.values():
            _, tmsg = timer.setdt()
            msg = msg + '\n' + tmsg

        return True, msg
    timer = _timers.get(target, None)
    if timer is None:
        return False, f'Timer {target} not found'
    return timer.setdt(newdt)


def step(recovery_time=0):
    ''' Increment the time of this clock with one timestep, plus a possible
        recovery time increment if the simulation is lagging and real-time
        running is enabled.
        Returns a floating-point representation of the new simulation time,
        and the actual timestep. '''
    recovery_time = min(Decimal(recovery_time), MAX_RECOVERY_FAC * _clock.dt)
    _clock.t += _clock.dt + recovery_time
    _clock.ft = float(_clock.t)
    for timer in _timers.values():
        timer.step()

    return _clock.ft, _clock.fdt + float(recovery_time)


def preupdate():
    ''' Update function executed before traffic update.'''
    for fun in preupdate_funs.values():
        fun.trigger()


def update():
    ''' Update function executed after traffic update.'''
    for fun in update_funs.values():
        fun.trigger()


def reset():
    ''' Reset function executed when simulation is reset.'''
    # Call plugin reset for plugins that have one
    for fun in reset_funs.values():
        fun.trigger()

    # Reset the simulation clock.
    _clock.t = Decimal('0.0')
    _clock.dt = Decimal(repr(settings.simdt))
    _clock.ft = 0.0
    _clock.fdt = float(_clock.dt)
    for timer in _timers.values():
        timer.reset()


class Timer:
    ''' Timer class for simulation-time periodic functions. '''
    @classmethod
    def maketimer(cls, name, dt):
        ''' Create and return a new timer if none with the given name exists.
            Return existing timer if present. '''
        return _timers.get(name.upper(), cls(name, dt))

    def __init__(self, name, dt):
        self.name = name
        self.dt_default = Decimal(repr(dt))
        self.dt_requested = self.dt_default
        self.dt_act = self.dt_default
        self.rel_freq = 0
        self.counter = 0
        self.tprev = _clock.t
        self.setdt()

        # Add self to dictionary of timers
        _timers[name.upper()] = self

    def reset(self):
        ''' Reset all simulation timers to their default time interval. '''
        self.dt_requested = self.dt_default
        self.dt_act = self.dt_default
        self.rel_freq = 0
        self.counter = 0
        self.tprev = _clock.t
        self.setdt()

    def setdt(self, dt=None):
        ''' Set the update interval of this timer. '''
        # setdt is called without arguments if the base dt has changed
        # In this case, check if our dt is still ok.
        if dt:
            # Store the requested dt separately: is used to update actual dt
            # for when the simulation dt is changed
            self.dt_requested = Decimal(repr(dt))

        # Calculate the relative frequency of the simulation with respect to this timer
        rel_freq = max(1, int(self.dt_requested // _clock.dt))
        # Update timer to next trigger point
        passed = self.rel_freq - self.counter
        self.counter = max(0, rel_freq - passed)
        self.rel_freq = rel_freq
        dtnew = self.rel_freq * _clock.dt
        if abs(self.dt_act - dtnew) < 0.0001:
            return True, self.name + ' dt is unchanged.'
        self.dt_act = dtnew
        if abs(self.dt_act - self.dt_requested) > 0.0001:
            return True, self.name + \
                f' dt set to {self.dt_act} to match integer multiple of base dt.'
        return True, self.name + f' dt set to {self.dt_act}'

    def step(self):
        ''' Step is called each base timestep to update this timer. '''
        self.counter = (self.counter or self.rel_freq) - 1

    def readynext(self):
        ''' Returns True if a time interval of this timer has passed. '''
        return self.counter == 0

    def elapsed(self):
        ''' Return the time elapsed since the last time this timer was triggered. '''
        elapsed = float(_clock.t - self.tprev)
        self.tprev = _clock.t
        return elapsed


class TimedFunction:
    ''' Wrapper object to hold (periodically) timed functions. '''
    def __init__(self, fun, name, dt=0, hook=''):
        self.trigger = None
        self.hook = hook
        if hasattr(fun, '__manualtimer__'):
            # If the passed function already has its own timer, use that one
            self.timer = fun.__manualtimer__
            self.name = self.timer.name
            self.callback = fun.__func__
        else:
            self.name = name
            # reset is a special case: doesn't need timer, and fun is called directly
            self.timer = None if hook == 'reset' else Timer.maketimer(name, dt)
            self.callback = fun

    @property
    def callback(self):
        ''' Callback pointing to the actual function triggered by this timer. '''
        return self._callback

    @callback.setter
    def callback(self, function):
        self._callback = function
        if self.timer is None:
            # Untimed functions are called directly
            self.trigger = function
        elif 'dt' in signature(function).parameters:
            # Timed functions that accept dt as argument
            self.trigger = self.call_timeddt
        else:
            # Timed functions without arguments
            self.trigger = self.call_timed
        if not inspect.ismethod(function):
            function.__timedfun__ = self

    def ismanual(self):
        ''' Returns true if this is a manually-triggered timed function. '''
        # This is a manually-triggered timed function if self.hook is empty.
        return not self.hook

    def call_timeddt(self):
        ''' Wrapper method to call timed functions that accept dt as argument. '''
        if self.timer.counter == 0:
            self._callback(dt=float(self.timer.dt_act))

    def call_timed(self):
        ''' Wrapper method to call timed functions. '''
        if self.timer.counter == 0:
            self._callback()

    def notimplemented(self, *args, **kwargs):
        ''' This function is called when a (derived) class is selected that doesn't
            provide the timed function originally passed by the base class. '''
        pass


def timed_function(fun=None, name='', dt=0, manual=False, hook='', timer=None):
    ''' Decorator to turn a function into a (periodically) timed function. '''
    def deco(fun):
        # Return original function if it is already wrapped
        if hasattr(fun, '__timedfun__') or (hasattr(fun, '__manualtimer__') and not hook):
            return fun
        # Generate a name if none is provided
        if name == '':
            if inspect.ismethod(fun):
                if inspect.isclass(fun.__self__):
                    # classmethods
                    tname = f'{fun.__self__.__name__}.{fun.__name__}'
                else:
                    # instance methods
                    tname = f'{fun.__self__.__class__.__name__}.{fun.__name__}'
            else:
                tname = f'{fun.__module__}.{fun.__name__}'
        else:
            tname = name

        if manual:
            manualtimer = timer or Timer.maketimer(tname, dt)
            if 'dt' in signature(fun).parameters:
                def wrapper(*args, **kwargs):
                    if manualtimer.counter == 0:
                        fun(*args, **kwargs, dt=float(manualtimer.dt_act))
            else:
                def wrapper(*args, **kwargs):
                    if manualtimer.counter == 0:
                        fun(*args, **kwargs)
            wrapper.__manualtimer__ = manualtimer
            wrapper.__func__ = fun
            return wrapper
        # Add automatically-triggered function to appropriate dict if not there yet.
        if hook == 'preupdate' and tname not in preupdate_funs:
            preupdate_funs[tname] = TimedFunction(fun, tname, dt, hook)
        elif hook == 'reset' and tname not in reset_funs:
            reset_funs[tname] = TimedFunction(fun, tname, dt, hook)
        elif tname not in update_funs:
            update_funs[tname] = TimedFunction(fun, tname, dt, 'update')
        return fun
    # Allow both @timed_function and @timed_function(args)
    return deco if fun is None else deco(fun)
