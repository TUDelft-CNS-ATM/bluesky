''' Node test '''
from collections.abc import Collection
import zmq
import msgpack
import bluesky as bs
from bluesky import stack
from bluesky.core import Entity, Signal
from bluesky.network import context as ctx
from bluesky.network.subscriber import Subscription
from bluesky.network.npcodec import encode_ndarray, decode_ndarray
from bluesky.network.common import genid, asbytestr, seqidx2id, seqid2idx, MSG_SUBSCRIBE, MSG_UNSUBSCRIBE, GROUPID_NOGROUP, GROUPID_CLIENT, GROUPID_SIM, GROUPID_DEFAULT, IDLEN


# Register settings defaults
bs.settings.set_variable_defaults(recv_port=11000, send_port=11001)


class Node(Entity):
    def __init__(self, group_id=None):
        super().__init__()
        self.node_id = genid(group_id or GROUPID_NOGROUP)
        self.group_id = asbytestr(group_id or GROUPID_NOGROUP)[:len(self.node_id)-1]
        self.server_id = self.node_id[:-1] + seqidx2id(0)
        self.act_id = None
        self.nodes = set()
        self.servers = set()

        zmqctx = zmq.Context.instance()
        self.sock_recv = zmqctx.socket(zmq.SUB)
        self.sock_send = zmqctx.socket(zmq.XPUB)
        self.poller = zmq.Poller()

        # Tell bluesky that this client will manage the network I/O
        bs.net = self

        # Subscribe to subscriptions that were already made before constructing
        # this node
        for sub in Subscription.subscriptions.values():
            sub.subscribe_all()

        # Signals
        self.node_added = Signal('node-added')
        self.node_removed = Signal('node-removed')
        self.server_added = Signal('server-added')
        self.server_removed = Signal('server-removed')

    def connect(self, hostname=None, recv_port=None, send_port=None, protocol='tcp'):
        ''' Connect node to a server.

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
        # Register this node by subscribing to targeted messages
        self.subscribe('',  '', to_group=self.node_id)

    def close(self):
        ''' Close all network connections. '''
        self.poller.unregister(self.sock_recv)
        self.poller.unregister(self.sock_send)
        self.sock_recv.close()
        self.sock_send.close()
        zmq.Context.instance().destroy()


    def update(self):
        ''' Node periodic update function.

            Periodically call this function to allow this node to receive
            and process data.
        '''
        # Check for incoming data
        self.receive()

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
            
                # Receive the message
                ctx.msg = sock.recv_multipart()
                if not ctx.msg:
                    # In the rare case that a message is empty, skip remaning processing
                    continue

                # Regular incoming data
                if sock == self.sock_recv:
                    ctx.topic = ctx.msg[0][IDLEN:-IDLEN].decode()
                    ctx.sender_id = ctx.msg[0][-IDLEN:]
                    pydata = msgpack.unpackb(ctx.msg[1], object_hook=decode_ndarray, raw=False)
                    sub = Subscription.subscriptions.get(ctx.topic, None) #or Subscription(ctx.topic, directedonly=True)
                    if sub is None:
                        print('No subscription known for', ctx.topic, 'on', self.node_id)
                        continue

                    # Unpack dict or list, skip empty string
                    if pydata == '':
                        sub.emit()
                    elif isinstance(pydata, dict):
                        sub.emit(**pydata)
                    elif isinstance(pydata, (list, tuple)):
                        sub.emit(*pydata)
                    else:
                        sub.emit(pydata)
                    ctx.msg = ctx.topic = ctx.sender_id = None

                elif sock == self.sock_send:
                    # This is an (un)subscribe message. If it's an id-only subscription
                    # this is also a registration message
                    if len(ctx.msg[0]) == IDLEN + 1:
                        sender_id = ctx.msg[0][1:]
                        sequence_idx = seqid2idx(sender_id[-1])
                        if sender_id[0] in (GROUPID_SIM, GROUPID_NOGROUP):
                            # This is an initial simulation node subscription
                            if ctx.msg[0][0] == MSG_SUBSCRIBE:
                                if sequence_idx > 0:
                                    self.nodes.add(sender_id)
                                    if sender_id != self.node_id:
                                        self.node_added.emit(sender_id)
                                    continue
                                elif sequence_idx == 0:
                                    self.servers.add(sender_id)
                                    self.server_added.emit(sender_id)

                            elif ctx.msg[0][0] == MSG_UNSUBSCRIBE:
                                if sequence_idx > 0:
                                    self.nodes.discard(sender_id)
                                    self.node_removed.emit(sender_id)
                                elif sequence_idx == 0:
                                    self.servers.discard(sender_id)
                                    self.server_removed.emit(sender_id)

        except zmq.ZMQError:
            return False

    def send(self, topic: str, data: str|Collection='', to_group: int|str|bytes=''):
        btopic = asbytestr(topic)
        bto_group = asbytestr(to_group or stack.sender() or '')
        self.sock_send.send_multipart(
            [
                bto_group.ljust(IDLEN, b'*') + btopic + self.node_id,
                msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
            ]
        )

    def subscribe(self, topic, from_group: int|str|bytes=GROUPID_DEFAULT, to_group: int|str|bytes='', actonly=False):
        ''' Subscribe to a topic.

            Arguments:
            - topic: The name of the topic to subscribe to
            - from_id: The id of the node from which to receive the topic (optional)
            - to_group: The group mask that this topic is sent to (optional)
            - actonly: Set to true if you only want to receive this topic from
              the active node. This only has effect when this Node is a Client.
        '''
        sub = None
        if topic:
            sub = Subscription(topic)
            sub.actonly = (sub.actonly or actonly)
            actonly = sub.actonly
            if (from_group, to_group) in sub.subs:
                # Subscription already active. Just return Subscription object
                return sub
            sub.subs.add((from_group, to_group))

        self._subscribe(topic, from_group, to_group, actonly)

        # Messages coming in that match this subscription will be emitted using a 
        # subscription signal
        return sub

    def unsubscribe(self, topic, from_group: int|str|bytes=GROUPID_DEFAULT, to_group: int|str|bytes=''):
        ''' Unsubscribe from a topic.

            Arguments:
            - topic: The name of the stream to unsubscribe from.
            - from_id: When subscribed to data from a specific node: The id of the node
            - to_group: The group mask that this topic is sent to (optional)
        '''
        if topic:
            Subscription(topic).subs.discard((from_group, to_group))
        self._unsubscribe(topic, from_group, to_group)

    def _subscribe(self, topic, from_group: int|str|bytes=GROUPID_DEFAULT, to_group: int|str|bytes='', actonly=False):
        if from_group == GROUPID_DEFAULT:
            from_group = GROUPID_CLIENT
        btopic = asbytestr(topic)
        bfrom_group = asbytestr(from_group)
        bto_group = asbytestr(to_group)
        self.sock_recv.setsockopt(zmq.SUBSCRIBE, bto_group.ljust(IDLEN, b'*') + btopic + bfrom_group)

    def _unsubscribe(self, topic, from_group: int|str|bytes=GROUPID_DEFAULT, to_group: int|str|bytes=''):
        if from_group == GROUPID_DEFAULT:
            from_group = GROUPID_CLIENT
        btopic = asbytestr(topic)
        bfrom_group = asbytestr(from_group)
        bto_group = asbytestr(to_group)
        self.sock_recv.setsockopt(zmq.UNSUBSCRIBE, bto_group.ljust(IDLEN, b'*') + btopic + bfrom_group)

    def addnodes(self, count=1, *node_ids):
        ''' Tell the server to add 'count' nodes. 
        
            If provided, create these nodes with the specified node ids.
        '''
        self.send('ADDNODES', dict(count=count, node_ids=node_ids), self.server_id)