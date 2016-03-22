try:
    # Try Qt5 first
    from PyQt5.QtCore import QThread, QObject
except ImportError:
    # Else PyQt4 imports
    from PyQt4.QtCore import QThread, QObject


class Thread(QThread):
    def __init__(self, worker):
        super(Thread, self).__init__()
        self.worker = worker
        worker.moveToThread(self)
        self.started.connect(worker.doWork)

    def start(self, prio):
        super(Thread, self).start()
        self.setPriority(prio)

    def quit(self):
        # Stop work in the thread
        self.worker.stop()

        # Quit the thread
        super(Thread, self).quit()
        print 'Ok.'


class ThreadManager(QObject):
    tm_instance = None

    @staticmethod
    def instance():
        return ThreadManager.tm_instance

    @staticmethod
    def currentThreadIsActive():
        return ThreadManager.tm_instance.active_node is Thread.currentThread()

    def __init__(self, parent=None):
        super(ThreadManager, self).__init__(parent)
        self.nodes       = []
        self.active_node = None

        if ThreadManager.tm_instance is not None:
            print 'Warning: a ThreadManager already exists!'
        else:
            ThreadManager.tm_instance = self

    def getSenderID(self):
        try:
            return self.nodes.index(self.sender())
        except:
            print 'ThreadManager: Error in retrieving event sender nodeid'
            return -1

    def setActiveNode(self, nodeid):
        if nodeid < len(self.nodes):
            self.active_node = self.nodes[nodeid]

    def startThread(self, worker_object, prio=Thread.HighestPriority):
        newnode = Thread(worker_object)
        self.nodes.append(newnode)
        newnode.start(prio)

        # Set new node as the active node
        self.active_node = newnode
