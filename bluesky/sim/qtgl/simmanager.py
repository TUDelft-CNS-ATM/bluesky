try:
    # Try Qt5 first
    from PyQt5.QtCore import pyqtSignal, QCoreApplication as qapp
except ImportError:
    # Else PyQt4 imports
    from PyQt4.QtCore import pyqtSignal, QCoreApplication as qapp
from copy import deepcopy
from thread import ThreadManager
from simulation import Simulation
from simevents import SimStateEventType, BatchEventType, BatchEvent, StackTextEvent


class SimulationManager(ThreadManager):
    # Signals
    nodes_changed = pyqtSignal(int)

    def __init__(self, navdb, parent=None):
        super(SimulationManager, self).__init__(parent)
        print 'Initializing multi-threaded simulation'
        self.navdb = navdb
        self.scentime = []
        self.scencmd  = []

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
        print 'Closing Gui'
        qapp.quit()

    def getSimObjectList(self):
        if len(self.nodes) == 1:
            return self.nodes[0].worker
        ret = []
        for node in self.nodes:
            ret.append(node.worker)
        return ret

    def getActiveSimTarget(self):
        return self.active_node.worker

    def event(self, event):
        if event.type() == SimStateEventType:
            if event.state == event.init:
                self.nodes_changed.emit(self.getSenderID())
            elif event.state == event.end:
                if len(self.scencmd) == 0:
                    if len(self.nodes) == 1:
                        self.quit()
                else:
                    # Find the scenario starts
                    scenidx  = [i for i in range(len(self.scencmd)) if self.scencmd[i][:4] == 'SCEN']
                    scenidx.append(len(self.scencmd))
                    start = scenidx[0]
                    end   = scenidx[1]
                    # Send a new scenario to the finished sim thread
                    qapp.postEvent(self.sender().worker,
                    BatchEvent(deepcopy(self.scentime[start:end]), deepcopy(self.scencmd[start:end])))

            return True
        elif event.type() == BatchEventType:
            self.scentime = event.scentime
            self.scencmd  = event.scencmd
            # Find the scenario starts
            scenidx  = [i for i in range(len(self.scencmd)) if self.scencmd[i][:4] == 'SCEN']
            # Check if the batch list contains scenarios
            if len(scenidx) == 0:
                qapp.postEvent(qapp.instance(), StackTextEvent(disptext='No scenarios defined in batch file!'))
                return True

            # Determine and start the required number of nodes
            reqd_nnodes = min(len(scenidx), self.max_nnodes)
            if reqd_nnodes > len(self.nodes):
                for n in range(len(self.nodes), reqd_nnodes):
                    self.addNode()
            # Distribute initial batch of tasks over the available nodes
            for s in range(min(reqd_nnodes, len(self.nodes))):
                start = scenidx[s]
                end   = len(self.scentime) if s+1 == len(scenidx) else scenidx[s+1]
                qapp.postEvent(self.nodes[s].worker,
                    BatchEvent(deepcopy(self.scentime[start:end]), deepcopy(self.scencmd[start:end])))

            # Delete the scenarios that were sent in the initial batch
            del self.scentime[0:end]
            del self.scencmd[0:end]

        return True
