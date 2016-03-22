try:
    # Try Qt5 first
    from PyQt5.QtCore import pyqtSignal
except ImportError:
    # Else PyQt4 imports
    from PyQt4.QtCore import pyqtSignal

from thread import ThreadManager
from simulation import Simulation
from simevents import SimStateEventType


class SimulationManager(ThreadManager):
    # Signals
    nodes_changed = pyqtSignal(int)

    def __init__(self, navdb, parent=None):
        super(SimulationManager, self).__init__(parent)
        self.navdb = navdb

    def addNode(self):
        worker = Simulation(self.navdb)
        self.startThread(worker)

    def quit(self):
        print 'Stopping Threads...'
        # Tell each thread to quit
        for n in range(len(self.nodes)):
            print 'Stopping node %d:' % n,
            self.nodes[n].quit()

        # Wait for all threads to finish
        for node in self.nodes:
            node.wait()
        print 'Done'

    def getActiveSimTarget(self):
        return self.active_node.worker.eventTarget()

    def event(self, event):
        if event.type() == SimStateEventType:
            if event.state == event.init:
                self.nodes_changed.emit(self.getSenderID())
            elif event.state == event.end:
                pass
                # self.closeAllWindows()
            return True
        return True
