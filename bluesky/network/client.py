''' BlueSky client base class. '''
import os
import zmq
import msgpack
import bluesky
from bluesky.tools import Signal
from bluesky.network.discovery import Discovery
from bluesky.network.npcodec import encode_ndarray, decode_ndarray


class Client(object):
    def __init__(self, actnode_topics=b''):
        ctx = zmq.Context.instance()
        self.event_io = ctx.socket(zmq.DEALER)
        self.stream_in = ctx.socket(zmq.SUB)
        self.poller = zmq.Poller()
        self.host_id = b''
        self.client_id = b'\x00' + os.urandom(4)
        self.sender_id = b''
        self.servers = dict()
        self.act = b''
        self.actroute = []
        self.acttopics = actnode_topics
        self.discovery = None

        # Signals
        self.nodes_changed = Signal()
        self.server_discovered = Signal()
        self.signal_quit = Signal()
        self.event_received = Signal()
        self.stream_received = Signal()

        # Tell bluesky that this client will manage the network I/O
        bluesky.net = self

    def start_discovery(self):
        if not self.discovery:
            self.discovery = Discovery(self.client_id)
            self.poller.register(self.discovery.handle, zmq.POLLIN)
            self.discovery.send_request()

    def stop_discovery(self):
        if self.discovery:
            self.poller.unregister(self.discovery.handle)
            self.discovery = None

    def get_hostid(self):
        return self.host_id

    def sender(self):
        return self.sender_id

    def event(self, name, data, sender_id):
        ''' Default event handler for Client. Override this function for added
            functionality. '''
        self.event_received.emit(name, data, sender_id)

    def stream(self, name, data, sender_id):
        ''' Default stream handler for Client. Override this function for added
            functionality. '''
        self.stream_received.emit(name, data, sender_id)

    def actnode_changed(self, newact):
        ''' Default actnode change handler for Client. Override or monkey-patch this function
            to implement actual actnode change handling. '''
        print('Client active node changed.')

    def subscribe(self, streamname, node_id=b''):
        ''' Subscribe to a stream. '''
        self.stream_in.setsockopt(zmq.SUBSCRIBE, streamname + node_id)

    def unsubscribe(self, streamname, node_id=b''):
        ''' Unsubscribe from a stream. '''
        self.stream_in.setsockopt(zmq.UNSUBSCRIBE, streamname + node_id)

    def connect(self, hostname='localhost', event_port=0, stream_port=0, protocol='tcp'):
        conbase = '{}://{}'.format(protocol, hostname)
        econ = conbase + (':{}'.format(event_port) if event_port else '')
        scon = conbase + (':{}'.format(stream_port) if stream_port else '')
        self.event_io.setsockopt(zmq.IDENTITY, self.client_id)
        self.event_io.connect(econ)
        self.send_event(b'REGISTER')
        self.host_id = self.event_io.recv_multipart()[0]
        print('Client {} connected to host {}'.format(self.client_id, self.host_id))
        self.stream_in.connect(scon)

        self.poller.register(self.event_io, zmq.POLLIN)
        self.poller.register(self.stream_in, zmq.POLLIN)

    def receive(self, timeout=0):
        ''' Poll for incoming data from Server, and receive if available.
            Arguments:
            timeout: The polling timeout in milliseconds. '''
        try:
            socks = dict(self.poller.poll(timeout))
            if socks.get(self.event_io) == zmq.POLLIN:
                msg = self.event_io.recv_multipart()
                # Remove send-to-all flag if present
                if msg[0] == b'*':
                    msg.pop(0)
                route, eventname, data = msg[:-2], msg[-2], msg[-1]
                self.sender_id = route[0]
                route.reverse()
                pydata = msgpack.unpackb(data, object_hook=decode_ndarray, encoding='utf-8')
                if eventname == b'NODESCHANGED':
                    self.servers.update(pydata)
                    self.nodes_changed.emit(pydata)

                    # If this is the first known node, select it as active node
                    nodes_myserver = next(iter(pydata.values())).get('nodes')
                    if not self.act and nodes_myserver:
                        self.actnode(nodes_myserver[0])
                elif eventname == b'QUIT':
                    self.signal_quit.emit()
                else:
                    self.event(eventname, pydata, self.sender_id)

            if socks.get(self.stream_in) == zmq.POLLIN:
                msg = self.stream_in.recv_multipart()

                strmname = msg[0][:-5]
                sender_id = msg[0][-5:]
                pydata = msgpack.unpackb(msg[1], object_hook=decode_ndarray, encoding='utf-8')
                self.stream(strmname, pydata, sender_id)

            # If we are in discovery mode, parse this message
            if self.discovery and socks.get(self.discovery.handle.fileno()):
                dmsg = self.discovery.recv_reqreply()
                if dmsg.conn_id != self.client_id and dmsg.is_server:
                    self.server_discovered.emit(dmsg.conn_ip, dmsg.ports)
        except zmq.ZMQError:
            return False

    def _getroute(self, target):
        for srv in self.servers.values():
            if target in srv['nodes']:
                return srv['route']
        return None

    def actnode(self, newact=None):
        if newact:
            route = self._getroute(newact)
            if route is None:
                print('Error selecting active node (unknown node)')
                return None
            # Unsubscribe from previous node, subscribe to new one.
            if newact != self.act:
                for topic in self.acttopics:
                    if self.act:
                        self.unsubscribe(topic, self.act)
                    self.subscribe(topic, newact)
                self.actroute = route
                self.act = newact
                self.actnode_changed(newact)

        return self.act

    def addnodes(self, count=1):
        self.send_event(b'ADDNODES', count)

    def send_event(self, name, data=None, target=None):
        pydata = msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
        if not target:
            self.event_io.send_multipart(self.actroute + [self.act, name, pydata])
        elif target == b'*':
            self.event_io.send_multipart([target, name, pydata])
        else:
            self.event_io.send_multipart(self._getroute(target) + [target, name, pydata])
