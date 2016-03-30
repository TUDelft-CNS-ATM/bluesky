try:
    from PyQt5.QtCore import QObject, QEvent
except ImportError:
    from PyQt4.QtCore import QObject, QEvent

# Local imports
from simulation import Simulation
from timer import Timer

# import faulthandler
# faulthandler.enable()


def runNode(connection, navdb, nodeid, active_node):
    manager     = NodeManager(connection, nodeid, active_node)
    manager.sim = Simulation(manager, navdb)
    manager.sim.doWork()
    manager.close()


class NodeManager(QObject):
    def __init__(self, connection, nodeid, active_node):
        super(NodeManager, self).__init__()
        self.connection      = connection
        self.sim             = None
        self.timers          = []
        self.nodeid          = nodeid
        self.active_node     = active_node

    def close(self):
        self.connection.close()

    def processEvents(self):
        # Process incoming data, and send to sim
        while self.connection.poll():
            (event, eventtype) = self.connection.recv()
            # Data over pipes is pickled/unpickled, this causes problems with
            # inherited classes. Solution is to call the ancestor's init
            QEvent.__init__(event, eventtype)
            self.sim.event(event)

        # Process timers
        Timer.updateTimers()

    def sendEvent(self, event):
        # Send event to the main process
        self.connection.send((event, int(event.type())))

    def isActive(self):
        return (self.nodeid == self.active_node.value)
