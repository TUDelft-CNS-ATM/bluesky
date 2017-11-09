import zmq
import msgpack
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

    def get_hostid(self):
        return self.host_id

    def event(self, name, data, sender_id):
        ''' Default event handler for Client. Override or monkey-patch this function
            to implement actual event handling. '''
        print('Client {} received event {} from {}'.format(self.client_id, name, sender_id))

    def stream(self, name, data, sender_id):
        ''' Default stream handler for Client. Override or monkey-patch this function
            to implement actual stream handling. '''
        print('Client {} received stream {} from {}'.format(self.client_id, name, sender_id))

    def nodes_changed(self, data):
        ''' Default node change handler for Client. Override or monkey-patch this function
            to implement actual node change handling. '''
        print('Client received node change info.')

    def subscribe(self, streamname, node_id=b''):
        ''' Subscribe to a stream. '''
        self.stream_in.setsockopt(zmq.SUBSCRIBE, streamname + node_id)

    def unsubscribe(self, streamname, node_id=b''):
        ''' Unsubscribe from a stream. '''
        self.stream_in.setsockopt(zmq.UNSUBSCRIBE, streamname + node_id)

    def connect(self):
        self.event_io.connect('tcp://localhost:9000')
        self.send_event(b'REGISTER')
        data = self.event_io.recv_multipart()[-1]
        self.client_id = 256 * data[-2] + data[-1]
        self.host_id   = data[:5]
        print('Client {} connected to host {}'.format(self.client_id, self.host_id))
        self.stream_in.connect('tcp://localhost:9001')
        # self.stream_in.setsockopt(zmq.SUBSCRIBE, b'')

        self.poller.register(self.event_io, zmq.POLLIN)
        self.poller.register(self.stream_in, zmq.POLLIN)

    def receive(self):
        ''' Poll for incoming data from Manager, and receive if available. '''
        try:
            socks = dict(self.poller.poll(0))
            if socks.get(self.event_io) == zmq.POLLIN:
                res = self.event_io.recv_multipart()
                self.sender_id = res[0]
                name = res[1]
                data = msgpack.unpackb(res[2], object_hook=decode_ndarray, encoding='utf-8')
                if name == b'NODESCHANGED':
                    self.known_nodes.update(data)
                    self.nodes_changed(data)
                else:
                    self.event(name, data, self.sender_id)

            if socks.get(self.stream_in) == zmq.POLLIN:
                res = self.stream_in.recv_multipart()

                name      = res[0][:-8]
                sender_id = res[0][-8:]
                data      = msgpack.unpackb(res[1], object_hook=decode_ndarray, encoding='utf-8')
                self.stream(name, data, sender_id)
        except zmq.ZMQError:
            return False

    def addnodes(self, count=1):
        self.send_event(b'ADDNODES', count)

    def send_event(self, name, data=None, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        self.event_io.send_multipart([target or b'*', name, msgpack.packb(data, default=encode_ndarray, use_bin_type=True)])

    def send_stream(self, name, data):
        pass
