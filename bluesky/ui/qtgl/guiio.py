''' I/O Client implementation for the QtGL gui. '''
try:
    from PyQt5.QtCore import QTimer

except ImportError:
    from PyQt4.QtCore import QTimer

from bluesky.io import Client
from bluesky.tools import Signal

# Globals
_act    = b''
_client = Client()
_timer  = None

# Signals
nodes_changed      = Signal()
activenode_changed = Signal()
event_received     = _client.event_received
stream_received    = _client.stream_received

def sender():
    return _client.sender_id

def init():
    _client.connect()
    global _timer
    _timer = QTimer()
    _timer.timeout.connect(_client.receive)
    _timer.start(10)

def send_event(data, target=None):
    _client.send_event(data, target or _act)

def actnode(newact=None):
    if newact is not None:
        global _act
        _act = newact
        activenode_changed.emit(newact)
        # TODO: unsubscribe from previous node, subscribe to new one.
    return _act
