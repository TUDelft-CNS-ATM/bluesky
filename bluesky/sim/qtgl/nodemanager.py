try:
    from PyQt5.QtCore import QObject, QEvent
except ImportError:
    from PyQt4.QtCore import QObject, QEvent

from multiprocessing.connection import Client

# Local imports
from simulation import Simulation
from timer import Timer
from simevents import SetNodeIdType, SetActiveNodeType, AddNodeType
# import faulthandler
# faulthandler.enable()


def runNode():
    connection  = Client(('localhost', 6000), authkey='bluesky')
    manager     = NodeManager(connection)
    manager.sim = Simulation(manager)
    manager.sim.doWork()
    manager.close()
    print 'Node', manager.nodeid, 'stopped.'


class NodeManager(QObject):
    def __init__(self, connection):
        super(NodeManager, self).__init__()
        self.connection      = connection
        self.sim             = None
        self.timers          = []
        self.nodeid          = -1
        self.active          = True

    def close(self):
        self.connection.close()

    def processEvents(self):
        # Process incoming data, and send to sim
        while self.connection.poll():
            (eventtype, event) = self.connection.recv()
            if eventtype == SetNodeIdType:
                self.nodeid = event
            elif eventtype == SetActiveNodeType:
                self.active = event
            else:
                # Data over pipes is pickled/unpickled, this causes problems with
                # inherited classes. Solution is to call the ancestor's init
                QEvent.__init__(event, eventtype)
                self.sim.event(event)

        # Process timers
        Timer.updateTimers()

    def sendEvent(self, event):
        # Send event to the main process
        self.connection.send((int(event.type()), event))

    def addNodes(self, count):
        self.connection.send((AddNodeType, count))

    def isActive(self):
        return self.active


if __name__ == '__main__':
    runNode()
