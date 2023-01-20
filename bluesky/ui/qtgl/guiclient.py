''' I/O Client implementation for the QtGL gui. '''
try:
    from PyQt5.QtCore import QTimer
except ImportError:
    from PyQt6.QtCore import QTimer
import numpy as np

from bluesky.ui.qtgl.customevents import ACDataEvent, RouteDataEvent
from bluesky.network.client import Client
from bluesky.network import subscriber, context as ctx
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
        self.subscribe(b'PANZOOM').connect(self.event)

    def start_discovery(self):
        super().start_discovery()
        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self.discovery.send_request)
        self.discovery_timer.start(3000)

    def stop_discovery(self):
        self.discovery_timer.stop()
        self.discovery_timer = None
        super().stop_discovery()



    @subscriber
    def echo(self, text='', flags=None, sender_id=b''):
        ''' Overloaded Client.echo function. '''
        # If sender_id is None this is an echo command originating from the gui user, and therefore also meant for the active node
        sender_id = ctx.sender_id or self.act_id
        sender_data = self.get_nodedata(sender_id)
        sender_data.echo(text, flags)
        if sender_id == self.act_id:
            self.actnodedata_changed.emit(sender_id, sender_data, ('ECHOTEXT',))

    def event(self, *args, **data):
        sender_data = self.get_nodedata(ctx.sender_id)
        data_changed = []
        if ctx.topic == 'RESET':
            sender_data.clear_scen_data()
            data_changed = list(UPDATE_ALL)
        elif ctx.topic == 'PANZOOM':
            sender_data.panzoom(**data)
            data_changed.append('PANZOOM')
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
    def __init__(self, route=None):
        # Stack window
        self.echo_text = ''
        self.stackcmds = dict()
        self.stacksyn = dict()

        # Display pan and zoom
        self.pan = [0.0, 0.0]
        self.zoom = 1.0

        self.naircraft = 0
        self.acdata = ACDataEvent()
        self.routedata = RouteDataEvent()

        # Per-scenario data
        self.clear_scen_data()

        # Network route to this node
        self._route = route

    def setacdata(self, data):
        self.acdata = ACDataEvent(data)
        self.naircraft = len(self.acdata.lat)

    def setroutedata(self, data):
        self.routedata = RouteDataEvent(data)

    def clear_scen_data(self):
        self.naircraft = 0
        self.acdata = ACDataEvent()
        self.routedata = RouteDataEvent()

    def siminit(self, shapes, **kwargs):
        self.__dict__.update(kwargs)

    def panzoom(self, pan=None, zoom=None, ar=1, absolute=True):
        if pan:
            if absolute:
                self.pan  = list(pan)
            else:
                self.pan[0] += pan[0]
                self.pan[1] += pan[1]
        if zoom:
            self.zoom = zoom * (1.0 if absolute else self.zoom)

    def echo(self, text='', flags=0, sender_id=None):
        if text:
            self.echo_text += ('\n' + text)
