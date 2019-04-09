''' Simulation clock with guaranteed decimal precision. '''
from collections import OrderedDict
from types import SimpleNamespace
import bluesky as bs
from decimal import Decimal
from bluesky import settings


settings.set_variable_defaults(simdt=0.05)

# Data that the simulation clock needs to keep
_clock = SimpleNamespace(t=Decimal('0.0'), dt=Decimal(repr(settings.simdt)),
                         ft=0.0, fdt=settings.simdt)
_timers = OrderedDict()


def setdt(newdt=None, target='simdt'):
    ''' Set the timestep for the simulation clock.
        Returns a floating-point representation of the new timestep. '''
    if newdt is None:
        text = 'Simulation timesteps:\nbase dt = {}'.format(_clock.fdt)
        for name, timer in _timers.items():
            text += '\n{} = {}'.format(timer.name, timer.dt_act)
        return True, text
    if target == 'simdt':
        _clock.dt = Decimal(repr(newdt))
        _clock.fdt = float(_clock.dt)
        msg = 'Base dt set to {}'.format(_clock.dt)
        for name, timer in _timers.items():
            _, tmsg = timer.setdt()
            msg = msg + '\n' + tmsg

        return True, msg
    timer = _timers.get(target, None)
    if timer is None:
        return False, 'Timer {} not found'.format(target)
    
    return timer.setdt(newdt)


def step():
    ''' Increment the time of this clock with one timestep. 
        Returns a floating-point representation of the new simulation time. '''
    _clock.t += _clock.dt
    _clock.ft = float(_clock.t)
    for timer in _timers.values():
        timer.step()

    return _clock.ft, _clock.fdt


def reset():
    ''' Reset the simulation clock. '''
    _clock.t = Decimal('0.0')
    _clock.dt = Decimal(repr(settings.simdt))
    _clock.ft = 0.0
    _clock.fdt = float(_clock.dt)
    for timer in _timers.values():
        timer.reset()


class Timer:
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
        self.dt_requested = self.dt_default
        self.dt_act = self.dt_default
        self.rel_freq = 0
        self.counter = 0
        self.tprev = _clock.t
        self.setdt()

    def setdt(self, dt=None):
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
                ' dt set to {} to match integer multiple of base dt.'.format(self.dt_act)
        else:
            return True, self.name + ' dt set to {}'.format(self.dt_act)        

    def step(self):
        self.counter = (self.counter or self.rel_freq) - 1

    def readynext(self):
        return self.counter == 0

    def elapsed(self):
        elapsed = float(_clock.t - self.tprev)
        self.tprev = _clock.t
        return elapsed


def timed_function(name, dt=1.0):
    def decorator(fun):
        timer = Timer(name, dt)
        def wrapper(*args, **kwargs):
            if timer.readynext():
                return fun(*args, **kwargs, dt=float(timer.dt_act))
        return wrapper
    return decorator
