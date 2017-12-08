''' BlueSky client base class. '''
import os
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
        self.client_id   = b'\x00' + os.urandom(4)
        self.sender_id   = b''
        self.known_nodes = dict()

    def get_hostid(self):
        return self.host_id

    def event(self, name, data, sender_id, sender_route):
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
        self.event_io.setsockopt(zmq.IDENTITY, self.client_id)
        self.event_io.connect('tcp://localhost:9000')
        self.send_event(b'REGISTER')
        self.host_id = self.event_io.recv_multipart()[0]
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
                msg = self.event_io.recv_multipart()
                route, eventname, data = msg[:-2], msg[-2], msg[-1]
                self.sender_id = route[0]
                route.reverse()
                pydata = msgpack.unpackb(data, object_hook=decode_ndarray, encoding='utf-8')
                if eventname == b'NODESCHANGED':
                    self.known_nodes.update(pydata)
                    self.nodes_changed(pydata)
                else:
                    self.event(eventname, pydata, self.sender_id, route)

            if socks.get(self.stream_in) == zmq.POLLIN:
                msg = self.stream_in.recv_multipart()

                strmname  = msg[0][:-5]
                sender_id = msg[0][-5:]
                pydata    = msgpack.unpackb(msg[1], object_hook=decode_ndarray, encoding='utf-8')
                self.stream(strmname, pydata, sender_id)
        except zmq.ZMQError:
            return False

    def addnodes(self, count=1):
        self.send_event(b'ADDNODES', count)

    def send_event(self, name, data=None, target=None):
        route = target or [b'*']
        self.event_io.send_multipart(route + [name, msgpack.packb(data, default=encode_ndarray, use_bin_type=True)])

    def send_stream(self, name, data):
        pass
