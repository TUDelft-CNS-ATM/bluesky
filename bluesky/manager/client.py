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
        self.event_io.send('REGISTER')
        msg = self.event_io.recv()
        self.client_id = ord(msg[-1]) - 100
        self.host_id   = msg[:4]
        print('Client {} connected to host {}'.format(self.client_id, self.host_id))
        self.stream_in.setsockopt(zmq.SUBSCRIBE, b'')
        self.stream_in.connect('tcp://localhost:9001')

        self.poller.register(self.event_io, zmq.POLLIN)
        self.poller.register(self.stream_in, zmq.POLLIN)

    def receive(self):
        ''' Poll for incoming data from Manager, and receive if avaiable. '''
        try:
            socks = dict(self.poller.poll(0))
            if socks.get(self.event_io) == zmq.POLLIN:
                sender_id = self.event_io.recv()
                data      = self.event_io.recv_pyobj()
                self.event_received.emit(data, sender_id)

            if socks.get(self.stream_in) == zmq.POLLIN:
                stream_name = self.stream_in.recv()
                data        = self.stream_in.recv_pyobj()
                self.stream_received.emit(data, stream_name)
        except zmq.ZMQError:
            return False

    def send_event(self, data, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        self.event_io.send(target or '*', zmq.SNDMORE)
        self.event_io.send_pyobj(data)

    def send_stream(self, data, name):
        pass
