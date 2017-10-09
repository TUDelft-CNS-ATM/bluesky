''' I/O Client implementation for the QtGL gui. '''
from bluesky.manager import Client
from bluesky.tools import Signal


# Signals
nodes_changed      = Signal()
activenode_changed = Signal()

# Globals
_act    = ''
_client = None


def init():
    global _client
    _client = Client()
    _client.connect()

def send_event(data, target=None):
    _client.send_event(data, target or _act or '*')

def actnode(newact=None):
    if newact is not None:
        global _act
        _act = newact
        activenode_changed.emit(newact)
        # TODO: unsubscribe from previous node, subscribe to new one.
    return _act
