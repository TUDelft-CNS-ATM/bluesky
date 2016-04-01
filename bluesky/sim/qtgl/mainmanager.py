try:
    from PyQt5.QtCore import QObject, QEvent, QTimer, pyqtSignal, \
        QCoreApplication as qapp
except ImportError:
    from PyQt4.QtCore import QObject, QEvent, QTimer, pyqtSignal, \
        QCoreApplication as qapp

# Local imports
from simevents import SimStateEventType, SimQuitEventType, BatchEventType, \
    BatchEvent, StackTextEvent, SimQuitEvent, SetNodeIdType, SetActiveNodeType

import select
import sys
from subprocess import Popen
from multiprocessing.connection import Listener
Listener.fileno = lambda self: self._listener._socket.fileno()


class MainManager(QObject):
    instance      = None
    # Signals
    nodes_changed = pyqtSignal(int)

    def __init__(self):
        super(MainManager, self).__init__()
        print 'Initializing multi-process simulation'
        MainManager.instance = self
        self.scentime        = []
        self.scencmd         = []
        self.nodes           = []
        self.connections     = []
        self.activenode      = 0
        self.sender_id       = -1
        self.listener        = Listener(('localhost', 6000), authkey='bluesky')

    def receiveFromNodes(self):
        # First look for new connections
        r, w, e = select.select((self.listener, ), (), (), 0)
        if self.listener in r:
            print "Received connection request from new node"
            conn = self.listener.accept()
            # Send the node information about its nodeid
            nodeid = len(self.connections)
            conn.send((SetNodeIdType, nodeid))
            self.connections.append(conn)
            self.setActiveNode(nodeid)

        # Then process any data in the active connections
        for self.sender_id in range(len(self.connections)):
            conn = self.connections[self.sender_id]
            if conn is None:
                continue
            # Check for incoming events with poll
            while conn.poll():
                # Receive events that are waiting in the conn
                try:
                    (eventtype, event) = conn.recv()
                except:
                    continue
                # Data over connections is pickled/unpickled, this causes problems with
                # inherited classes. Solution is to call the ancestor's init
                QEvent.__init__(event, eventtype)

                # First check if this event is meant for the manager
                if event.type() == SimStateEventType:
                    if event.state == event.init:
                        self.nodes_changed.emit(self.sender_id)
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
                            # Send a new scenario to the finished sim process
                            conn.send((BatchEventType, BatchEvent(self.scentime[start:end], self.scencmd[start:end])))

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
                        if reqd_nnodes > len(self.nodes):
                            for n in range(len(self.nodes), reqd_nnodes):
                                self.addNode()
                        # Distribute initial batch of tasks over the available nodes
                        for s in range(min(reqd_nnodes, len(self.nodes))):
                            start = scenidx[s]
                            end   = len(self.scentime) if s+1 == len(scenidx) else scenidx[s+1]
                            self.connections[s].send((BatchEventType, BatchEvent(self.scentime[start:end], self.scencmd[start:end])))

                        # Delete the scenarios that were sent in the initial batch
                        del self.scentime[0:end]
                        del self.scencmd[0:end]
                else:
                    # The event is meant for the gui
                    qapp.sendEvent(qapp.instance(), event)

        # To avoid giving wrong information with getSenderID() when it is called outside
        # of this function, set sender_id to -1
        self.sender_id = -1

    def addNode(self):
        if len(self.connections) > 0:
            self.connections[self.activenode].send((SetActiveNodeType, False))
        p = Popen([sys.executable, 'BlueSky_qtgl.py', '--node'])
        self.nodes.append(p)

    def setActiveNode(self, nodeid):
        if nodeid < len(self.connections):
            if not nodeid == self.activenode:
                self.connections[self.activenode].send((SetActiveNodeType, False))
                self.activenode = nodeid
                self.connections[self.activenode].send((SetActiveNodeType, True))

    def getSenderID(self):
        return self.sender_id

    def start(self):
        timer           = QTimer(self)
        timer.timeout.connect(self.receiveFromNodes)
        timer.start(20)

        # Start the first simulation node on start
        self.addNode()

    def quit(self):
        print 'Stopping simulation processes...'
        # Tell each process to quit
        quitevent = (SimQuitEventType, SimQuitEvent())
        for n in range(len(self.connections)):
            print 'Stopping node %d:' % n,
            self.connections[n].send(quitevent)

        # Wait for all threads to finish
        # for node in self.nodes:
        #     node.wait()
        print 'Done'
        print 'Closing Gui'
        qapp.quit()

    def event(self, event):
        # Only send custom events to the active node
        if event.type() >= 1000:
            self.connections[self.activenode].send((int(event.type()), event))
        return True
