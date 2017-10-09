from bluesky import settings
if settings.is_sim:
    from bluesky.manager.timer import Timer
else:
    from bluesky.manager.client import Client

_manager = None


def init():
    ''' Initialization of the manager module. '''
    global _manager, _io
    if settings.is_sim:
        from bluesky.manager.node import Node
        _manager = Node()
    else:
        from bluesky.manager.main import Manager
        _manager = Manager()


def send_event(event, target=None):
    ''' Send event-based data to target.
        If there is no target ID, the event is sent to all clients. '''
    _manager.send_event(event, target)


def send_stream(data, name):
    ''' Send named stream data. Receivers can subscribe to this data using
        this name. '''
    _manager.send_stream(data, name)

def start():
    _manager.start()
