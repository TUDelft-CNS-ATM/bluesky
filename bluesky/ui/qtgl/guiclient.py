''' I/O Client implementation for the QtGL gui. '''
try:
    from PyQt5.QtCore import QTimer
except ImportError:
    from PyQt6.QtCore import QTimer

from bluesky.network.client import Client
from bluesky.network import context as ctx
from bluesky.core import Signal


# Globals
UPDATE_ALL = ['SHAPE', 'TRAILS', 'CUSTWPT', 'PANZOOM', 'ECHOTEXT', 'ROUTEDATA']


class GuiClient(Client):
    def __init__(self):
        super().__init__()
        self.nodedata = dict()
        self.ref_nodedata = nodeData()
        self.discovery_timer = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)

        # Signals
        self.actnodedata_changed = Signal('actnode-changed')

        # Connect to signals. TODO: needs revision
        Signal('SIMSTATE').connect(self.event)
        self.subscribe('RESET').connect(self.event)

    def start_discovery(self):
        super().start_discovery()
        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self.discovery.send_request)
        self.discovery_timer.start(3000)

    def stop_discovery(self):
        self.discovery_timer.stop()
        self.discovery_timer = None
        super().stop_discovery()

    def event(self, *args, **data):
        sender_data = self.get_nodedata(ctx.sender_id)
        data_changed = []
        if ctx.topic == 'RESET':
            data_changed = list(UPDATE_ALL)
        elif ctx.topic == 'SIMSTATE':
            sender_data.siminit(**data)
            data_changed = list(UPDATE_ALL)
        # else:
        #     super().event(ctx.topic, data, ctx.sender_id)

        if ctx.sender_id == self.act_id and data_changed:
            self.actnodedata_changed.emit(ctx.sender_id, sender_data, data_changed)

    def actnode_changed(self, newact):
        self.actnodedata_changed.emit(newact, self.get_nodedata(newact), UPDATE_ALL)

    def get_nodedata(self, nodeid=None):
        nodeid = nodeid or self.act_id
        if not nodeid:
            return self.ref_nodedata

        data = self.nodedata.get(nodeid)
        if not data:
            # If this is a node we haven't addressed yet: create dataset and
            # request node settings
            self.nodedata[nodeid] = data = nodeData()
            self.send(b'GETSIMSTATE', to_group=nodeid)

        return data


class nodeData:
    def __init__(self):
        # Stack window
        self.stackcmds = dict()
        self.stacksyn = dict()

    def siminit(self, shapes, **kwargs):
        self.__dict__.update(kwargs)
