from thread import ThreadManager
from simulation import Simulation


class SimulationManager(ThreadManager):
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
        print 'SimulationManager: received event'
        return True
