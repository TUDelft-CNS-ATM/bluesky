''' I/O Client implementation for the QtGL gui. '''
from bluesky.io import Client
from bluesky.tools import Signal

# Globals
_act    = ''
_client = Client()

# Signals
nodes_changed      = Signal()
activenode_changed = Signal()
event_received     = _client.event_received
stream_received    = _client.stream_received

def sender():
    return _client.sender_id

def init():
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
