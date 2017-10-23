import zmq
from bluesky.tools import Signal


class Client(object):
    def __init__(self):
        ctx = zmq.Context.instance()
        self.event_io  = ctx.socket(zmq.DEALER)
        self.stream_in = ctx.socket(zmq.SUB)
        self.poller    = zmq.Poller()
        self.host_id   = ''
        self.client_id = 0

        # Signals
        self.event_received  = Signal()
        self.stream_received = Signal()

    def connect(self):
        self.event_io.connect('tcp://localhost:9000')
        self.event_io.send(b'REGISTER')
        msg = self.event_io.recv()
        self.client_id = 256 * msg[-2] + msg[-1]
        self.host_id   = msg[:5]
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
                sender_id = self.event_io.recv()
                data      = self.event_io.recv_pyobj()
                print('received event data')
                print(data)
                self.event_received.emit(data, sender_id)

            if socks.get(self.stream_in) == zmq.POLLIN:
                nameandid   = self.stream_in.recv()
                stream_name = nameandid[:-8]
                sender_id   = nameandid[-8:]
                data        = self.stream_in.recv_pyobj()
                print('received stream data {}'.format(stream_name))
                print(data)
                self.stream_received.emit(data, stream_name, sender_id)
        except zmq.ZMQError:
            return False

    def addnodes(self, count=1):
        self.event_io.send(bytearray((count,)), zmq.SNDMORE)
        self.event_io.send(b'ADDNODES')

    def send_event(self, data, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        self.event_io.send(bytearray(target or '*', 'ascii'), zmq.SNDMORE)
        self.event_io.send_pyobj(data)

    def send_stream(self, data, name):
        pass
