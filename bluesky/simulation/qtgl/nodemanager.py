""" NodeManager manages I/O with the simulation processes on the Sim side. """
from multiprocessing.connection import Client
try:
    from PyQt5.QtCore import QEvent
except ImportError:
    from PyQt4.QtCore import QEvent

# Local imports
import bluesky as bs
from .timer import Timer
from .simevents import SetNodeIdType, SetActiveNodeType, AddNodeType
# import faulthandler
# faulthandler.enable()

connection = None
timers     = []
nodeid     = -1
active     = True


def run():
    global connection
    connection = Client(('localhost', 6000), authkey=b'bluesky')
    bs.sim.doWork()
    connection.close()
    print('Node', nodeid, 'stopped.')


def close():
    connection.close()


def processEvents():
    global nodeid, active
    # Process incoming data, and send to sim
    while connection.poll():
        (eventtype, event) = connection.recv()
        if eventtype == SetNodeIdType:
            nodeid = event
        elif eventtype == SetActiveNodeType:
            active = event
        else:
            # Data over pipes is pickled/unpickled, this causes problems with
            # inherited classes. Solution is to call the ancestor's init
            QEvent.__init__(event, eventtype)
            bs.sim.event(event)

    # Process timers
    Timer.updateTimers()


def sendEvent(event):
    # Send event to the main process
    connection.send((int(event.type()), event))


def addNodes(count):
    connection.send((AddNodeType, count))


def isActive():
    return active


if __name__ == '__main__':
    run()
