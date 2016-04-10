try:
    from PyQt5.QtCore import QObject, QEvent, QTimer, pyqtSignal, \
        QCoreApplication as qapp
except ImportError:
    from PyQt4.QtCore import QObject, QEvent, QTimer, pyqtSignal, \
        QCoreApplication as qapp

# Local imports
from simevents import SimStateEventType, SimQuitEventType, BatchEventType, \
    BatchEvent, StackTextEvent, SimQuitEvent, SetNodeIdType, \
    SetActiveNodeType, AddNodeType

import select
import sys
from subprocess import Popen
from multiprocessing.connection import Listener
from multiprocessing import cpu_count
Listener.fileno = lambda self: self._listener._socket.fileno()


class MainManager(QObject):
    instance           = None
    # Signals
    nodes_changed      = pyqtSignal(str, tuple, int)
    activenode_changed = pyqtSignal(tuple, int)

    @classmethod
    def sender(cls):
        return cls.instance.sender_id

    def __init__(self):
        super(MainManager, self).__init__()
        print 'Initializing multi-process simulation'
        MainManager.instance = self
        self.scentime        = []
        self.scencmd         = []
        self.connections     = []
        self.localnodes      = []
        self.hosts           = dict()
        self.max_nnodes      = cpu_count()
        self.activenode      = 0
        self.sender_id       = -1
        self.stopping        = False
        self.listener        = Listener(('localhost', 6000), authkey='bluesky')

    def receiveFromNodes(self):
        # Only look for incoming data if we're not quitting
        if self.stopping:
            return

        # First look for new connections
        r, w, e = select.select((self.listener, ), (), (), 0)
        if self.listener in r:
            conn = self.listener.accept()
            address = self.listener.last_accepted[0]
            if address in self.hosts:
                nodeid = self.hosts[address]
            else:
                nodeid = (len(self.hosts), 0)
            # Store host number and its number of nodes
            self.hosts[address] = (nodeid[0], nodeid[1] + 1)
            # Send the node information about its nodeid
            connidx = len(self.connections)
            conn.send((SetNodeIdType, nodeid))
            self.connections.append((conn, nodeid))
            self.nodes_changed.emit(address, nodeid, connidx)
            self.setActiveNode(connidx)

        # Then process any data in the active connections
        for connidx in range(len(self.connections)):
            conn = self.connections[connidx]
            if conn[0] is None or conn[0].closed:
                continue

            # Check for incoming events with poll
            while conn[0].poll():
                # Receive events that are waiting in the conn
                try:
                    (eventtype, event) = conn[0].recv()
                except:
                    continue

                # Sender id is connection index and node id
                self.sender_id = (connidx, conn[1])

                if eventtype == AddNodeType:
                    # This event only consists of an int: the number of nodes to add
                    for i in range(event):
                        self.addNode()
                    continue
                # Data over connections is pickled/unpickled, this causes problems with
                # inherited classes. Solution is to call the ancestor's init
                QEvent.__init__(event, eventtype)

                # First check if this event is meant for the manager
                if event.type() == SimStateEventType:
                    if event.state == event.init:
                        # Set received state to end to enable sending of new batch scenario
                        if len(self.localnodes) > 1:
                            event.state = event.end

                    if event.state == event.end:
                        if len(self.scencmd) == 0:
                            if len(self.localnodes) == 1:
                                # Quit the main loop. Afterwards, manager will also quit
                                qapp.instance().quit()

                        else:
                            # Find the scenario starts
                            scenidx  = [i for i in range(len(self.scencmd)) if self.scencmd[i][:4] == 'SCEN']
                            scenidx.append(len(self.scencmd))
                            start = scenidx[0]
                            end   = scenidx[1]
                            # Send a new scenario to the finished sim process
                            conn[0].send((BatchEventType, BatchEvent(self.scentime[start:end], self.scencmd[start:end])))

                            # Delete the scenarios that were sent in the initial batch
                            del self.scentime[0:end]
                            del self.scencmd[0:end]

                elif event.type() == BatchEventType:
                    self.scentime = event.scentime
                    self.scencmd  = event.scencmd
                    # Find the scenario starts
                    scenidx  = [i for i in range(len(self.scencmd)) if self.scencmd[i][:4] == 'SCEN']
                    # Check if the batch list contains scenarios
                    if len(scenidx) == 0:
                        qapp.sendEvent(qapp.instance(), StackTextEvent(disptext='No scenarios defined in batch file!'))
                    else:
                        # Determine and start the required number of nodes
                        reqd_nnodes = min(len(scenidx), self.max_nnodes)
                        if reqd_nnodes > len(self.localnodes):
                            for n in range(len(self.localnodes), reqd_nnodes):
                                self.addNode()

                else:
                    # The event is meant for the gui
                    qapp.sendEvent(qapp.instance(), event)

        # To avoid giving wrong information with getSenderID() when it is called outside
        # of this function, set sender_id to -1
        self.sender_id = -1

    def addNode(self):
        if len(self.connections) > 0:
            self.connections[self.activenode][0].send((SetActiveNodeType, False))
        p = Popen([sys.executable, 'BlueSky_qtgl.py', '--node'])
        self.localnodes.append(p)

    def setActiveNode(self, connidx):
        if connidx < len(self.connections):
            self.activenode_changed.emit(self.connections[connidx][1], connidx)
            if not connidx == self.activenode:
                self.connections[self.activenode][0].send((SetActiveNodeType, False))
                self.activenode = connidx
                self.connections[self.activenode][0].send((SetActiveNodeType, True))

    def start(self):
        timer           = QTimer(self)
        timer.timeout.connect(self.receiveFromNodes)
        timer.start(20)

        # Start the first simulation node on start
        self.addNode()

    def stop(self):
        print 'Stopping simulation processes...'
        self.stopping = True
        # Tell each process to quit
        quitevent = (SimQuitEventType, SimQuitEvent())
        print 'Stopping nodes:'
        for n in range(len(self.connections)):
            self.connections[n][0].send(quitevent)

        # Wait for all nodes to finish
        for n in self.localnodes:
            n.wait()

        for n in range(len(self.connections)):
            self.connections[n][0].close()
        print 'Done.'

        # Quit the main loop
        qapp.quit()

    def event(self, event):
        # Only send custom events to the active node
        if event.type() >= 1000:
            self.connections[self.activenode][0].send((int(event.type()), event))
        return True
