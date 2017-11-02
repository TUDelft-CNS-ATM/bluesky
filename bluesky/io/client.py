import zmq
import msgpack
from bluesky.tools import Signal
from bluesky.io.npcodec import encode_ndarray, decode_ndarray


class Client(object):
    def __init__(self):
        ctx = zmq.Context.instance()
        self.event_io    = ctx.socket(zmq.DEALER)
        self.stream_in   = ctx.socket(zmq.SUB)
        self.poller      = zmq.Poller()
        self.host_id     = b''
        self.client_id   = 0
        self.sender_id   = b''
        self.known_nodes = dict()

        # Signals
        self.nodes_changed   = Signal()
        self.event_received  = Signal()
        self.stream_received = Signal()

    def connect(self):
        self.event_io.connect('tcp://localhost:9000')
        self.send_event(b'REGISTER')
        data = self.event_io.recv_multipart()[-1]
        self.client_id = 256 * data[-2] + data[-1]
        self.host_id   = data[:5]
        print('Client {} connected to host {}'.format(self.client_id, self.host_id))
        self.stream_in.connect('tcp://localhost:9001')
        self.stream_in.setsockopt(zmq.SUBSCRIBE, b'')

        self.poller.register(self.event_io, zmq.POLLIN)
        self.poller.register(self.stream_in, zmq.POLLIN)

    def receive(self):
        ''' Poll for incoming data from Manager, and receive if avaiable. '''
        try:
            socks = dict(self.poller.poll(0))
            if socks.get(self.event_io) == zmq.POLLIN:
                res = self.event_io.recv_multipart()
                self.sender_id = res[0]
                name = res[1]
                data = msgpack.unpackb(res[2], object_hook=decode_ndarray, encoding='utf-8')
                if name == b'NODESCHANGED':
                    self.known_nodes.update(data)
                    self.nodes_changed.emit(data)

                self.event_received.emit(name, data, self.sender_id)

            if socks.get(self.stream_in) == zmq.POLLIN:
                res = self.stream_in.recv_multipart()

                name      = res[0][:-8]
                sender_id = res[0][-8:]
                data      = msgpack.unpackb(res[1], object_hook=decode_ndarray, encoding='utf-8')
                self.stream_received.emit(name, data, sender_id)
        except zmq.ZMQError:
            return False

    def addnodes(self, count=1):
        self.send_event(b'ADDNODES', count)

    def send_event(self, name, data=None, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        self.event_io.send_multipart([target or b'*', name, msgpack.packb(data, default=encode_ndarray, use_bin_type=True)])

    def send_stream(self, name, data):
        pass
