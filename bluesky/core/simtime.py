''' Simulation clock with guaranteed decimal precision. '''
import time
from collections.abc import Collection
from types import SimpleNamespace
from decimal import Decimal

from bluesky import settings


# Register settings defaults
settings.set_variable_defaults(simdt=0.05)

MAX_RECOVERY_FAC = 4

# Data that the simulation clock needs to keep
_clock = SimpleNamespace(t=Decimal('0.0'), dt=Decimal(repr(settings.simdt)),
                         ft=0.0, fdt=settings.simdt)


def setdt(newdt=None, target='simdt'):
    ''' Set the timestep for the simulation clock.
        Returns a floating-point representation of the new timestep. '''
    if newdt is None:
        text = f'Simulation timesteps:\nbase dt = {_clock.fdt}'
        for timer in Timer.timers():
            text += f'\n{timer.name} = {timer.dt_act}'
        return True, text
    if target == 'simdt':
        _clock.dt = Decimal(repr(newdt))
        _clock.fdt = float(_clock.dt)
        msg = f'Base dt set to {_clock.dt}'
        for timer in Timer.timers():
            _, tmsg = timer.setdt()
            msg = msg + '\n' + tmsg

        return True, msg
    timer = Timer.gettimer(target)
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
    for timer in Timer.timers():
        timer.step()

    return _clock.ft, _clock.fdt + float(recovery_time)


def reset():
    ''' Reset function executed when simulation is reset.'''
    # Reset the simulation clock.
    _clock.t = Decimal('0.0')
    _clock.dt = Decimal(repr(settings.simdt))
    _clock.ft = 0.0
    _clock.fdt = float(_clock.dt)
    for timer in Timer.timers():
        timer.reset()


class TimerMeta(type):
    __timers__: dict[str, 'Timer'] = dict()

    def __call__(cls, name, dt):
        timer = TimerMeta.__timers__.get(name)
        if timer is None:
            TimerMeta.__timers__[name] = timer = super().__call__(name, dt)
        return timer

    @classmethod
    def timers(cls) -> Collection['Timer']:
        return cls.__timers__.values()

    @classmethod
    def gettimer(cls, name) -> 'Timer|None':
        return cls.__timers__.get(name)


class Timer(metaclass=TimerMeta):
    ''' Timer class for simulation-time periodic functions. '''

    def __init__(self, name, dt):
        self.name = name
        self.dt_default = Decimal(repr(dt))
        self.dt_requested = self.dt_default
        self.dt_act = self.dt_default
        self.rel_freq = 0
        self.counter = 0
        self.tprev = _clock.t
        self.readynext = True

        self.setdt()

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
        self.readynext = self.counter == 0

    def elapsed(self):
        ''' Return the time elapsed since the last time this timer was triggered. '''
        elapsed = float(_clock.t - self.tprev)
        self.tprev = _clock.t
        return elapsed
