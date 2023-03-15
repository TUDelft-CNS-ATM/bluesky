''' I/O Client implementation for the QtGL gui. '''
try:
    from PyQt5.QtCore import QTimer
except ImportError:
    from PyQt6.QtCore import QTimer

from bluesky.network.client import Client
from bluesky.network import context as ctx
from bluesky.core import Signal


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

    def start_discovery(self):
        super().start_discovery()
        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self.discovery.send_request)
        self.discovery_timer.start(3000)

    def stop_discovery(self):
        self.discovery_timer.stop()
        self.discovery_timer = None
        super().stop_discovery()



class nodeData:
    def __init__(self):
        # Stack window TODO: naar clientstack
        self.stackcmds = dict()
        self.stacksyn = dict()
