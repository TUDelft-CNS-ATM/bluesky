try:
    from PyQt5.QtCore import QObject, pyqtSignal
except ImportError:
    from PyQt4.QtCore import QObject, pyqtSignal

import time


class Timer(QObject):
    timeout = pyqtSignal()
    timers  = None

    def __init__(self):
        super(Timer, self).__init__()
        self.interval = 0
        self.t_next   = 0.0

    def start(self, interval):
        if Timer.timers is None:
            Timer.timers = []

        Timer.timers.append(self)
        self.interval = float(interval) * 1e-3
        self.t_next   = time.time() + self.interval

    @classmethod
    def updateTimers(cls):
        tcur = time.time()
        for timer in cls.timers:
            if tcur >= timer.t_next:
                timer.timeout.emit()
                timer.t_next += timer.interval
