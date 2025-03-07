from collections import defaultdict
from typing import Union, Collection
import zmq
import msgpack

import bluesky as bs
from bluesky import stack
from bluesky.core import Entity, Signal
from bluesky.stack.clientstack import process
from bluesky.traffic.trafficproxy import TrafficProxy
from bluesky.network import context as ctx
from bluesky.network.npcodec import encode_ndarray, decode_ndarray
from bluesky.network.subscriber import Subscription
from bluesky.network.common import genid, asbytestr, seqid2idx, MSG_SUBSCRIBE, MSG_UNSUBSCRIBE, GROUPID_NOGROUP, GROUPID_CLIENT, GROUPID_SIM, GROUPID_DEFAULT, IDLEN


# Register settings defaults
bs.settings.set_variable_defaults(recv_port=11000, send_port=11001)


class Client(Entity):
    def __init__(self, group_id=GROUPID_CLIENT):
        super().__init__()
        self.group_id = asbytestr(group_id)
        self.client_id = genid(self.group_id)
        self.act_id = None
        self.acttopics = defaultdict(set)
        self.nodes = set()
        self.servers = set()
        self.discovery = None

        zmqctx = zmq.Context.instance()
        self.sock_recv = zmqctx.socket(zmq.SUB)
        self.sock_send = zmqctx.socket(zmq.XPUB)
        self.poller = zmq.Poller()

        # Tell bluesky that this client will manage the network I/O
        bs.net = self
        # And create a proxy object for easy traffic data access
        bs.traf = TrafficProxy()

        # Subscribe to subscriptions that were already made before constructing
        # this node
        for sub in Subscription.subscriptions.values():
            sub.subscribe_all()

        # Signals
        self.actnode_changed = Signal('actnode-changed')
        self.node_added = Signal('node-added')
        self.node_removed = Signal('node-removed')
        self.server_added = Signal('server-added')
        self.server_removed = Signal('server-removed')

        # If no other object is taking care of this, let this client act as screen object as well
        if not bs.scr:
            bs.scr = self

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
        self.subscribe('', '', to_group=self.client_id)

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
                    sub = Subscription.subscriptions.get(ctx.topic, None)# or Subscription(ctx.topic, directedonly=True)
                    if sub is None:
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
                                    self.node_added.emit(sender_id)
                                    if not self.act_id:
                                        self.actnode(sender_id)
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

    def send(self, topic: str, data: Union[str, Collection]='', to_group: str=''):
        btopic = asbytestr(topic)
        btarget = asbytestr(to_group or stack.sender() or self.act_id or '')
        self.sock_send.send_multipart(
            [
                btarget.ljust(IDLEN, b'*') + btopic + self.client_id,
                msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
            ]
        )

    def subscribe(self, topic, from_group=GROUPID_DEFAULT, to_group='', actonly=False):
        ''' Subscribe to a topic.

            Arguments:
            - topic: The name of the topic to subscribe to
            - from_id: The id of the node from which to receive the topic (optional)
            - to_group: The group mask that this topic is sent to (optional)
            - actonly: Set to true if you only want to receive this topic from
              the active node.
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

    def unsubscribe(self, topic, from_group=GROUPID_DEFAULT, to_group=''):
        ''' Unsubscribe from a topic.

            Arguments:
            - topic: The name of the stream to unsubscribe from.
            - from_id: When subscribed to data from a specific node: The id of the node
            - to_group: The group mask that this topic is sent to (optional)
        '''
        if topic:
            Subscription(topic).subs.discard((from_group, to_group))
        self._unsubscribe(topic, from_group, to_group)

    def _subscribe(self, topic, from_group=GROUPID_DEFAULT, to_group='', actonly=False):
        if from_group == GROUPID_DEFAULT:
            from_group = GROUPID_SIM
            if actonly:
                self.acttopics[topic].add(to_group)
                if self.act_id is not None:
                    bfrom_group = self.act_id
                else:
                    return
        btopic = asbytestr(topic)
        bfrom_group = asbytestr(from_group)
        bto_group = asbytestr(to_group)

        self.sock_recv.setsockopt(zmq.SUBSCRIBE, bto_group.ljust(IDLEN, b'*') + btopic + bfrom_group)

    def _unsubscribe(self, topic, from_group=GROUPID_DEFAULT, to_group=''):
        if from_group == GROUPID_DEFAULT:
            from_group = GROUPID_SIM
        btopic = asbytestr(topic)
        bfrom_group = asbytestr(from_group)
        bto_group = asbytestr(to_group)
        if from_group == GROUPID_DEFAULT and topic in self.acttopics:
            self.acttopics[topic].discard(to_group)
            if self.act_id is not None:
                bfrom_group = self.act_id
            else:
                return
        self.sock_recv.setsockopt(zmq.UNSUBSCRIBE, bto_group.ljust(IDLEN, b'*') + btopic + bfrom_group)

    def actnode(self, newact=None):
        ''' Set the new active node, or return the current active node. '''
        if newact:
            if newact not in self.nodes:
                print('Error selecting active node (unknown node)')
                return None
            # Unsubscribe from previous node, subscribe to new one.
            if newact != self.act_id:
                for topic, groupset in self.acttopics.items():
                    for to_group in groupset:
                        if self.act_id:
                            self._unsubscribe(topic, self.act_id, to_group)
                        self._subscribe(topic, newact, to_group)
                self.act_id = newact
                self.actnode_changed.emit(newact)

        return self.act_id

    def addnodes(self, count=1, *node_ids, server_id=None):
        ''' Tell the specified server to add 'count' nodes. 
        
            If provided, create these nodes with the specified node ids.

            If no server_id is specified, the corresponding server of the
            currently-active node is targeted.
        '''
        self.send(b'ADDNODES', dict(count=count, node_ids=node_ids), server_id or genid(self.act_id[:-1], seqidx=0))
