""" Node encapsulates the sim process, and manages process I/O. """
from threading import Thread
import zmq
import msgpack
from bluesky import stack
from bluesky.tools import Timer
from bluesky.io.npcodec import encode_ndarray, decode_ndarray


class Node(object):
    def __init__(self):
        self.host_id = b''
        self.node_id = b''
        self.running = True
        ctx = zmq.Context.instance()
        self.event_io = ctx.socket(zmq.DEALER)
        self.stream_out = ctx.socket(zmq.PUB)

    def event(self, eventname, eventdata, sender_id):
        ''' Event data handler. Reimplemented in Simulation. '''
        print('Received {} data from {}'.format(eventname, sender_id))

    def step(self):
        ''' Perform one iteration step. Reimplemented in Simulation. '''
        pass

    def start(self):
        ''' Starting of main loop. '''
        # Final Initialization
        # Initialization of sockets.
        self.event_io.connect('tcp://localhost:10000')
        self.stream_out.connect('tcp://localhost:10001')

        # Start communication, and receive this node's ID
        self.send_event(b'REGISTER')
        self.node_id = self.event_io.recv_multipart()[-1]
        self.host_id = self.node_id[:5]
        print('Node started, id={}'.format(self.node_id))

        # run() implements the main loop
        self.run()

    def quit(self):
        ''' Quit the simulation process. '''
        self.running = False

    def run(self):
        ''' Start the main loop of this node. '''
        while self.running:
            # Get new events from the I/O thread
            # while self.event_io.poll(0):
            if self.event_io.getsockopt(zmq.EVENTS) & zmq.POLLIN:
                res = self.event_io.recv_multipart()
                sender_id = res[0]
                name = res[1]
                if name == b'QUIT':
                    self.quit()
                else:
                    data = msgpack.unpackb(res[2], object_hook=decode_ndarray, encoding='utf-8')
                    self.event(name, data, sender_id)
            # Perform a simulation step
            self.step()

            # Process timers
            Timer.update_timers()

    def addnodes(self, count=1):
        self.send_event(b'ADDNODES', count)

    def send_event(self, name, data=None, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        self.event_io.send_multipart([stack.sender() or b'*', name, msgpack.packb(data, default=encode_ndarray, use_bin_type=True)])

    def send_stream(self, name, data):
        self.stream_out.send_multipart([name + self.node_id, msgpack.packb(data, default=encode_ndarray, use_bin_type=True)])
