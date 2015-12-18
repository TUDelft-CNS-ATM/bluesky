try:
    # Try Qt5 first
    from PyQt5.QtCore import QThread
except ImportError:
    # Else PyQt4 imports
    from PyQt4.QtCore import QThread


class Thread(QThread):
    def __init__(self, worker_object):
        super(Thread, self).__init__()
        worker_object.moveToThread(self)
        self.started.connect(worker_object.doWork)

    def start(self, prio):
        super(Thread, self).start()
        self.setPriority(prio)
