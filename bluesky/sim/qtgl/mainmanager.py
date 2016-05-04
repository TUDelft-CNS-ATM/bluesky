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


def split_scenarios(scentime, scencmd):
    start = 0
    for i in xrange(1, len(scencmd) + 1):
        if i == len(scencmd) or scencmd[i][:4] == 'SCEN':
            scenname = scencmd[start].split()[1].strip()
            yield (scenname, scentime[start:i], scencmd[start:i])
            start = i


class MainManager(QObject):
    instance           = None
    # Signals
    nodes_changed      = pyqtSignal(str, tuple, int)
    activenode_changed = pyqtSignal(tuple, int)

    @classmethod
    def sender(cls):
        return cls.instance.sender_id

    @classmethod
    def actnode(cls):
        return cls.instance.activenode

    def __init__(self):
        super(MainManager, self).__init__()
        print 'Initializing multi-process simulation'
        MainManager.instance = self
        self.scenarios       = []
        self.connections     = []
        self.localnodes      = []
        self.hosts           = dict()
        self.max_nnodes      = cpu_count()
        self.activenode      = 0
        self.sender_id       = None
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
            self.connections.append([conn, nodeid, 0])
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
                    # Save the state together with the connection object
                    conn[2] = event.state
                    if event.state == event.end:
                        # Quit the main loop. Afterwards, manager will also quit
                        qapp.instance().quit()

                    elif event.state == event.init or event.state == event.hold:
                        if len(self.scenarios) > 0:
                            self.sendScenario(conn)

                elif event.type() == BatchEventType:
                    self.scenarios = [scen for scen in split_scenarios(event.scentime, event.scencmd)]
                    # Check if the batch list contains scenarios
                    if len(self.scenarios) == 0:
                        qapp.sendEvent(qapp.instance(), StackTextEvent(disptext='No scenarios defined in batch file!'))

                    else:
                        qapp.sendEvent(qapp.instance(), StackTextEvent(disptext='Found %d scenarios in batch' % len(self.scenarios)))
                        # Available nodes (nodes that are in init or hold mode):
                        av_nodes = [n for n in range(len(self.connections)) if self.connections[n][2] in [0, 2]]
                        for i in range(min(len(av_nodes), len(self.scenarios))):
                            self.sendScenario(self.connections[i])
                        # If there are still scenarios left, determine and start the required number of local nodes
                        reqd_nnodes = min(len(self.scenarios), max(0, self.max_nnodes - len(self.localnodes)))
                        for n in range(reqd_nnodes):
                            self.addNode()

                else:
                    # The event is meant for the gui
                    qapp.sendEvent(qapp.instance(), event)

        # To avoid giving wrong information with sender() when it is called outside
        # of this function, set sender_id to None
        self.sender_id = None

    def addNode(self):
        if len(self.connections) > 0:
            self.connections[self.activenode][0].send((SetActiveNodeType, False))
        p = Popen([sys.executable, 'BlueSky_qtgl.py', '--node'])
        self.localnodes.append(p)

    def sendScenario(self, conn):
        # Send a new scenario to the target sim process
        scen = self.scenarios[0]
        qapp.sendEvent(qapp.instance(), StackTextEvent(disptext='Starting scenario ' + scen[0] + ' on node (%d, %d)' % conn[1]))
        conn[0].send((BatchEventType, BatchEvent(scen[1], scen[2])))

        # Delete the scenario from the list
        del self.scenarios[0]

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

    def event(self, event):
        # Only send custom events to the active node
        if event.type() >= 1000:
            self.connections[self.activenode][0].send((int(event.type()), event))
        return True
