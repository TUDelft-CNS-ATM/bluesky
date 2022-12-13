import zmq
import msgpack
import bluesky as bs
from bluesky import stack
from bluesky.core import Entity, Signal
from bluesky.stack.clientstack import process
from bluesky.network.npcodec import encode_ndarray, decode_ndarray
from bluesky.network.discovery import Discovery
from bluesky.network.subscription import Subscription
from bluesky.network.common import genid, asbytestr, seqid2idx, MSG_SUBSCRIBE, MSG_UNSUBSCRIBE, GROUPID_NOGROUP, GROUPID_CLIENT, GROUPID_SIM, IDLEN


# Register settings defaults
bs.settings.set_variable_defaults(recv_port=9000, send_port=9001)


class Client(Entity):
    def __init__(self, group_id=GROUPID_CLIENT):
        self.group_id = asbytestr(group_id)
        self.client_id = genid(self.group_id)
        self.sender_id = b''
        self.act_id = b''
        self.acttopics = set()
        self.nodes = set()
        self.servers = set()
        self.running = True
        self.discovery = None

        ctx = zmq.Context.instance()
        self.sock_recv = ctx.socket(zmq.SUB)
        self.sock_send = ctx.socket(zmq.XPUB)
        self.poller = zmq.Poller()        

        # Subscribe to subscriptions that were already made before constructing
        # this node
        for sub in Subscription.subscriptions.values():
            self.subscribe(sub.topic, sub.from_id, sub.to_group, sub.actonly)

        # Signals
        self.nodes_changed = Signal('nodes_changed')
        self.server_discovered = Signal('server_discovered')
        self.signal_quit = Signal('quit')

        # Tell bluesky that this client will manage the network I/O
        bs.net = self
        # If no other object is taking care of this, let this client act as screen object as well
        if not bs.scr:
            bs.scr = self

    def quit(self):
        self.running = False

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

    def connect(self, hostname=None, recv_port=None, send_port=None, protocol='tcp'):
        ''' Connect client to a server.

            Arguments:
            - hostname: Network name or ip of the server to connect to
            - recv_port: Network port to use for incoming communication
            - send_port: Network port to use for outgoing communication
            - protocol: Network protocol to use
        '''
        conbase = f'{protocol}://{hostname or "localhost"}'
        rcon = conbase + f':{recv_port or bs.settings.recv_port}'
        scon = conbase + f':{send_port or bs.settings.send_port}'
        self.sock_recv.connect(rcon)
        self.sock_send.connect(scon)
        self.poller.register(self.sock_recv, zmq.POLLIN)
        self.poller.register(self.sock_send, zmq.POLLIN)
        # Register this client by subscribing to targeted messages
        self.subscribe(b'', to_group=self.client_id)

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
            events = dict(self.poller.poll(timeout))

            # The socket with incoming data
            for sock, event in events.items():
                if event != zmq.POLLIN:
                    # The event does not refer to incoming data: skip for now
                    continue

                # If we are in discovery mode, parse this message
                if self.discovery and sock == self.discovery.handle.fileno():
                    dmsg = self.discovery.recv_reqreply()
                    if dmsg.conn_id != self.client_id and dmsg.is_server:
                        self.server_discovered.emit(dmsg.conn_ip, dmsg.ports)
                    continue
            
                # Receive the message
                msg = sock.recv_multipart()
                if not msg:
                    # In the rare case that a message is empty, skip remaning processing
                    continue
            
                if sock == self.sock_recv:
                    topic, self.sender_id = msg[0][IDLEN:-IDLEN], msg[0][-IDLEN:]
                    pydata = msgpack.unpackb(msg[1], object_hook=decode_ndarray, raw=False)
                    sub = Subscription.subscriptions.get(topic) or Subscription(topic)
                    sub.emit(pydata)

                elif sock == self.sock_send:
                    # This is an (un)subscribe message. If it's an id-only subscription
                    # this is also a registration message
                    if len(msg[0]) == IDLEN + 1:
                        sender_id = msg[0][1:]
                        sequence_idx = seqid2idx(sender_id[-1])
                        if sender_id[0] in (GROUPID_SIM, GROUPID_NOGROUP):
                            # This is an initial simulation node subscription
                            if msg[0][0] == MSG_SUBSCRIBE:
                                if sequence_idx > 0:
                                    self.nodes.add(sender_id)
                                    if not self.act_id:
                                        self.act_id = sender_id
                                elif sequence_idx == 0:
                                    self.servers.add(sender_id)
                                else:
                                    return

                            elif msg[0][0] == MSG_UNSUBSCRIBE:
                                if sequence_idx > 0:
                                    self.nodes.discard(sender_id)
                                elif sequence_idx == 0:
                                    self.servers.discard(sender_id)
                                else:
                                    return
                            self.nodes_changed.emit(self.nodes, self.servers)

        except zmq.ZMQError:
            return False

    def send(self, topic, data='', to_group=''):
        topic = asbytestr(topic)
        to_group = asbytestr(to_group or stack.sender() or self.act_id or GROUPID_SIM)
        self.sock_send.send_multipart(
            [
                to_group.ljust(IDLEN, b'*') + topic + self.client_id,
                msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
            ]
        )

    def subscribe(self, topic, from_id='', to_group=GROUPID_CLIENT, actonly=False):
        ''' Subscribe to a topic.

            Arguments:
            - topic: The name of the topic to subscribe to
            - from_id: The id of the node from which to receive the topic (optional)
            - to_group: The group mask that this topic is sent to (optional)
            - actonly: Set to true if you only want to receive this topic from
              the active node.
        '''
        topic = asbytestr(topic)
        from_id = asbytestr(from_id)
        to_group = asbytestr(to_group)
        if actonly and not from_id and topic not in self.acttopics:
            self.acttopics.add(topic)
            from_id = self.act_id
        self.sock_recv.setsockopt(zmq.SUBSCRIBE, to_group.ljust(IDLEN, b'*') + topic + from_id)

        # Messages coming in that match this subscription will be emitted using a 
        # subscription signal
        if topic:
            return Subscription(topic, from_id, to_group, actonly)
    
    def unsubscribe(self, topic, from_id='', to_group=GROUPID_CLIENT):
        ''' Unsubscribe from a topic.

            Arguments:
            - topic: The name of the stream to unsubscribe from.
            - from_id: When subscribed to data from a specific node: The id of the node
            - to_group: The group mask that this topic is sent to (optional)
        '''
        topic = asbytestr(topic)
        from_id = asbytestr(from_id)
        to_group = asbytestr(to_group)
        if not to_group and topic in self.acttopics:
            self.acttopics.remove(topic)
            to_group = self.act_id
        self.sock_recv.setsockopt(zmq.UNSUBSCRIBE, to_group.ljust(IDLEN, b'*') + topic + from_id)

    def actnode(self, newact=None):
        ''' Set the new active node, or return the current active node. '''
        if newact:
            if newact not in self.nodes:
                print('Error selecting active node (unknown node)')
                return None
            # Unsubscribe from previous node, subscribe to new one.
            if newact != self.act_id:
                for topic in self.acttopics:
                    if self.act_id:
                        self.unsubscribe(topic, self.act_id)
                    self.subscribe(topic, newact)
                self.act_id = newact
                self.actnode_changed(newact)

        return self.act_id

    def actnode_changed(self, newact):
        ''' Default actnode change handler for Client. Override or monkey-patch this function
            to implement actual actnode change handling. '''
        print('Client active node changed.')

    def addnodes(self, count=1):
        ''' Tell the server to add 'count' nodes. '''
        self.send(b'ADDNODES', count)# TODO: get server_id in
