try:
    from PyQt5.QtCore import QObject, QEvent, QTimer, pyqtSignal, QCoreApplication as qapp
except ImportError:
    from PyQt4.QtCore import QObject, QEvent, QTimer, pyqtSignal, QCoreApplication as qapp

import multiprocessing as mp

# Local imports
from nodemanager import runNode
from simevents import SimStateEventType, SimQuitEventType, BatchEventType, BatchEvent, StackTextEvent, SimQuitEvent


class MainManager(QObject):
    instance      = None
    # Signals
    nodes_changed = pyqtSignal(int)

    def __init__(self, navdb):
        super(MainManager, self).__init__()
        print 'Initializing multi-process simulation'
        # mp.set_start_method('spawn')
        MainManager.instance = self
        self.navdb           = navdb
        self.scentime        = []
        self.scencmd         = []
        self.nodes           = []
        self.activenode      = mp.Value('i', 0)
        self.sender_id       = -1

    def receiveFromNodes(self):
        for self.sender_id in range(len(self.nodes)):
            pipe = self.nodes[self.sender_id][1]
            # Check for incoming events with poll
            while pipe.poll():
                # Receive events that are waiting in the pipe
                (event, eventtype) = pipe.recv()
                # Data over pipes is pickled/unpickled, this causes problems with
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
                            pipe.send(BatchEvent(self.scentime[start:end], self.scencmd[start:end]))

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
                            self.nodes[s][1].send(BatchEvent(self.scentime[start:end], self.scencmd[start:end]))

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
        parent_conn, child_conn = mp.Pipe()
        new_nodeid = len(self.nodes)
        p = mp.Process(target=runNode, args=(child_conn, self.navdb, new_nodeid, self.activenode))
        self.nodes.append((p, parent_conn))
        self.activenode.value = new_nodeid
        p.start()

    def setActiveNode(self, nodeid):
        if nodeid < len(self.nodes):
            self.activenode.value = nodeid

    def getSenderID(self):
        return self.sender_id

    def start(self):
        timer           = QTimer(self)
        timer.timeout.connect(self.receiveFromNodes)
        timer.start(20)

    def quit(self):
        print 'Stopping simulation processes...'
        # Tell each process to quit
        quitevent = (SimQuitEvent(), SimQuitEventType)
        for n in range(len(self.nodes)):
            print 'Stopping node %d:' % n,
            self.nodes[n][1].send(quitevent)

        # Wait for all threads to finish
        for node in self.nodes:
            node[0].join()
        print 'Done'
        print 'Closing Gui'
        qapp.quit()

    def event(self, event):
        # Only send custom events to the active node
        if event.type() >= 1000:
            self.nodes[self.activenode.value][1].send((event, int(event.type())))
        return True
