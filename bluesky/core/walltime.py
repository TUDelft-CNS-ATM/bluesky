""" BlueSky implementation of a timer that can periodically trigger functions."""
import time
from math import remainder
from bluesky.core.signal import Signal


class Timer:
    """ A timer can be used to periodically (wall-time) trigger functions."""

    # Data that the wall clock needs to keep
    __timers__ = []

    def __init__(self, interval: int):
        super().__init__()
        self.timeout  = Signal()
        self.interval = interval * 1000000
        self.t_next   = (time.time_ns() // self.interval + 1) * self.interval

        # Append self to known timers
        Timer.__timers__.append(self)

    @classmethod
    def update_timers(cls):
        ''' Update all timers. '''
        tcur = time.time_ns()
        for timer in cls.__timers__:
            if tcur >= timer.t_next:
                timer.timeout.emit()
                timer.t_next = (tcur // timer.interval + 1) * timer.interval
