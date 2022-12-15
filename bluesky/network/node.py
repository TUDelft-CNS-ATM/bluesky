''' Node test '''
import signal
import zmq
import msgpack
import bluesky as bs
from bluesky import stack
from bluesky.core import Entity
from bluesky.core.walltime import Timer
from bluesky.network.subscription import Subscription
from bluesky.network.npcodec import encode_ndarray, decode_ndarray
from bluesky.network.common import genid, seqidx2id, asbytestr, hex2bin, GROUPID_NOGROUP, GROUPID_CLIENT, GROUPID_SIM, IDLEN


# Register settings defaults
bs.settings.set_variable_defaults(recv_port=9000, send_port=9001)


class Node(Entity):
    def __init__(self, group_id=GROUPID_NOGROUP):
        self.node_id = genid(group_id)
        self.group_id = asbytestr(group_id)[:len(self.node_id)-1]
        self.server_id = self.node_id[:-1] + seqidx2id(0)
        self.sender_id = b''
        ctx = zmq.Context.instance()
        self.sock_recv = ctx.socket(zmq.SUB)
        self.sock_send = ctx.socket(zmq.PUB)
        self.poller = zmq.Poller()
        self.running = True
        signal.signal(signal.SIGINT, lambda *args: self.quit())

        # Subscribe to subscriptions that were already made before constructing
        # this node
        for sub in Subscription.subscriptions.values():
            sub.subscribe_all()

    def quit(self):
        ''' Quit the simulation process. '''
        self.running = False

    def stop(self):
        ''' Stack stop/quit command. '''
        # On a stack quit command, send quit signal to server to stop all
        # simulations in this group.
        self.send(b'QUIT', to_group=self.server_id)

    def connect(self):
        ''' Connect node to the BlueSky server. '''
        self.sock_recv.connect(f'tcp://localhost:{bs.settings.recv_port}')
        self.sock_send.connect(f'tcp://localhost:{bs.settings.send_port}')
        self.poller.register(self.sock_recv, zmq.POLLIN)
        # Register this node by subscribing to targeted messages
        self.subscribe(b'', to_group=self.node_id)

    def run(self):
        ''' Start the main loop of this node. '''
        while self.running:
            # Perform a simulation step
            self.update()
            bs.sim.update()
            # Update screen logic
            bs.scr.update()

        self.sock_recv.close()
        self.sock_send.close()
        zmq.Context.instance().destroy()

    def update(self):
        ''' Update timers and perform I/O. '''
        # Process timers
        Timer.update_timers()
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
                msg = sock.recv_multipart()
                if not msg:
                    # In the rare case that a message is empty, skip remaning processing
                    continue
            
                if sock == self.sock_recv:
                    topic, self.sender_id = msg[0][IDLEN:-IDLEN], msg[0][-IDLEN:]
                    pydata = msgpack.unpackb(msg[1], object_hook=decode_ndarray, raw=False)
                    sub = Subscription.subscriptions.get(topic) or Subscription(topic)
                    sub.emit(pydata)

        except zmq.ZMQError:
            return False

    def addnodes(self, count=1):
        self.send(b'ADDNODES', count, self.server_id)

    def send(self, topic, data='', to_group=''):
        topic = asbytestr(topic)
        to_group = asbytestr(to_group or stack.sender() or GROUPID_CLIENT)
        self.sock_send.send_multipart(
            [
                to_group.ljust(IDLEN, b'*') + topic + self.node_id,
                msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
            ]
        )

    def subscribe(self, topic, from_id='', to_group=None):
        ''' Subscribe to a topic.

            Arguments:
            - topic: The name of the topic to subscribe to
            - from_id: The id of the node from which to receive the topic (optional)
            - to_group: The group mask that this topic is sent to (optional)
        '''
        sub = None
        if topic:
            sub = Subscription(topic)
            if (from_id, to_group) in sub.subs:
                # Subscription already active. Just return Subscription object
                return sub
            sub.subs.append((from_id, to_group))

        self._subscribe(topic, from_id, to_group)

        # Messages coming in that match this subscription will be emitted using a 
        # subscription signal
        return sub

    def unsubscribe(self, topic, from_id='', to_group=None):
        ''' Unsubscribe from a topic.

            Arguments:
            - topic: The name of the stream to unsubscribe from.
            - from_id: When subscribed to data from a specific node: The id of the node
            - to_group: The group mask that this topic is sent to (optional)
        '''
        sub = None
        if topic and (from_id or to_group is not None):
            sub = Subscription(topic)
            sub.subs.discard((from_id, to_group))
        self._unsubscribe(topic, from_id, to_group)

    def _unsubscribe(self, topic, from_id='', to_group=None):
        ''' Unsubscribe from a topic.

            Arguments:
            - topic: The name of the stream to unsubscribe from.
            - from_id: When subscribed to data from a specific node: The id of the node
            - to_group: The group mask that this topic is sent to (optional)
        '''
        topic = asbytestr(topic)
        from_id = asbytestr(from_id)
        to_group = asbytestr(to_group or GROUPID_SIM)
        self.sock_recv.setsockopt(zmq.UNSUBSCRIBE, to_group.ljust(IDLEN, b'*') + topic + from_id)


    def _subscribe(self, topic, from_id='', to_group=None):
        topic = asbytestr(topic)
        from_id = asbytestr(from_id)
        to_group = asbytestr(to_group or GROUPID_SIM)
        self.sock_recv.setsockopt(zmq.SUBSCRIBE, to_group.ljust(IDLEN, b'*') + topic + from_id)