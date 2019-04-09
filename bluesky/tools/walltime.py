""" BlueSky implementation of a timer that can periodically trigger functions."""
import time
from collections import OrderedDict
from bluesky.tools.signal import Signal


class Timer(object):
    """ A timer can be used to periodically (wall-time) trigger functions."""

    # Data that the wall clock needs to keep
    timers = []


    def __init__(self):
        super(Timer, self).__init__()
        self.timeout  = Signal()
        self.interval = 0.0
        self.t_next   = 0.0

    def start(self, interval):
        Timer.timers.append(self)
        self.interval = float(interval) * 1e-3
        self.t_next   = time.time() + self.interval

    @classmethod
    def update_timers(cls):
        tcur = time.time()
        for timer in cls.timers:
            if tcur >= timer.t_next:
                timer.timeout.emit()
                timer.t_next += timer.interval
