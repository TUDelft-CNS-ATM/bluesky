''' BlueSky client base class. '''
import os
import zmq
import msgpack
import bluesky
from bluesky import settings
from bluesky.core import Signal
from bluesky.stack.clientstack import stack, process
from bluesky.network.discovery import Discovery
from bluesky.network.npcodec import encode_ndarray, decode_ndarray


class Client:
    ''' Base class for (GUI) clients of a BlueSky server. '''
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
        self.nodes_changed = Signal('nodes_changed')
        self.server_discovered = Signal('server_discovered')
        self.signal_quit = Signal('quit')
        self.event_received = Signal('event_received')
        self.stream_received = Signal('stream_received')

        # Tell bluesky that this client will manage the network I/O
        bluesky.net = self
        # If no other object is taking care of this, let this client act as screen object as well
        if not bluesky.scr:
            bluesky.scr = self

    def start_discovery(self):
        ''' Start UDP-based discovery of available BlueSky servers. '''
        if not self.discovery:
            self.discovery = Discovery(self.client_id)
            self.poller.register(self.discovery.handle, zmq.POLLIN)
            self.discovery.send_request()

    def stop_discovery(self):
        ''' Stop UDP-based discovery. '''
        if self.discovery:
            self.poller.unregister(self.discovery.handle)
            self.discovery = None

    def get_hostid(self):
        ''' Return the id of the host that this client is connected to. '''
        return self.host_id

    def sender(self):
        ''' Return the id of the sender of the most recent event. '''
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

    def subscribe(self, streamname, node_id=b'', actonly=False):
        ''' Subscribe to a stream.

            Arguments:
            - streamname: The name of the stream to subscribe to
            - node_id: The id of the node from which to receive the stream (optional)
            - actonly: Set to true if you only want to receive this stream from
              the active node.
        '''
        if actonly and not node_id and streamname not in self.acttopics:
            self.acttopics.append(streamname)
            node_id = self.act
        self.stream_in.setsockopt(zmq.SUBSCRIBE, streamname + node_id)

    def unsubscribe(self, streamname, node_id=b''):
        ''' Unsubscribe from a stream.

            Arguments:
            - streamname: The name of the stream to unsubscribe from.
            - node_id: ID of the specific node to unsubscribe from.
                       This is also used when switching active nodes.
        '''
        if not node_id and streamname in self.acttopics:
            self.acttopics.remove(streamname)
            node_id = self.act
        self.stream_in.setsockopt(zmq.UNSUBSCRIBE, streamname + node_id)

    def connect(self, hostname=None, event_port=None, stream_port=None, protocol='tcp'):
        ''' Connect client to a server.

            Arguments:
            - hostname: Network name or ip of the server to connect to
            - event_port: Network port to use for event communication
            - stream_port: Network port to use for stream communication
            - protocol: Network protocol to use
        '''
        conbase = f'{protocol}://{hostname or "localhost"}'
        econ = conbase + f':{event_port or settings.event_port}'
        scon = conbase + f':{stream_port or settings.stream_port}'
        self.event_io.setsockopt(zmq.IDENTITY, self.client_id)
        self.event_io.connect(econ)
        self.send_event(b'REGISTER')
        self.host_id = self.event_io.recv_multipart()[0]
        print(f'Client {self.client_id} connected to host {self.host_id}')
        self.stream_in.connect(scon)

        self.poller.register(self.event_io, zmq.POLLIN)
        self.poller.register(self.stream_in, zmq.POLLIN)

    def echo(self, text, flags=None, sender_id=None):
        ''' Default client echo function. Prints to console.
            Overload this function to process echo text in your GUI. '''
        print(text)

    def update(self):
        ''' Client periodic update function.

            Periodically call this function to allow client to receive and process data.
        '''
        self.receive()
        # Process any waiting stacked commands
        process()

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
                self.sender_id, *_, eventname, data = msg
                pydata = msgpack.unpackb(data, object_hook=decode_ndarray, raw=False)
                if eventname == b'STACK':
                    stack(pydata, sender_id=self.sender_id)
                elif eventname == b'ECHO':
                    self.echo(**pydata, sender_id=self.sender_id)
                elif eventname == b'NODESCHANGED':
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
                if self._getroute(sender_id) is None:
                    print('Client: Skipping stream data from unknown node')
                    return False
                pydata = msgpack.unpackb(msg[1], object_hook=decode_ndarray, raw=False)
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
        ''' Set the new active node, or return the current active node. '''
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
        ''' Tell the server to add 'count' nodes. '''
        self.send_event(b'ADDNODES', count)

    def send_event(self, name, data=None, target=None):
        ''' Send an event to one or all simulation node(s).

            Arguments:
            - name: Name of the event
            - data: Data to send as payload
            - target: Destination of this event. Event is sent to all nodes
              if * is specified as target.
        '''
        pydata = msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
        if not target:
            self.event_io.send_multipart(self.actroute + [self.act, name, pydata])
        elif target == b'*':
            self.event_io.send_multipart([target, name, pydata])
        else:
            rte = self._getroute(target)
            if rte is None:
                print(f'Client: Not sending event {name} to unknown target {target}')
                return
            self.event_io.send_multipart(rte + [target, name, pydata])
