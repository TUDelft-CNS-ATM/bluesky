""" Node encapsulates the sim process, and manages process I/O. """
from threading import Thread
import zmq
from bluesky import stack
from bluesky.tools import Timer


class IOThread(Thread):
    ''' Separate thread for node I/O. '''
    def run(self):
        ''' Implementation of the I/O loop. '''
        ctx = zmq.Context.instance()
        fe_event = ctx.socket(zmq.DEALER)
        fe_stream = ctx.socket(zmq.PUB)
        be_event = ctx.socket(zmq.PAIR)
        be_stream = ctx.socket(zmq.PAIR)
        fe_event.connect('tcp://localhost:10000')
        fe_stream.connect('tcp://localhost:10001')

        be_event.connect('inproc://event')
        be_stream.connect('inproc://stream')
        poller = zmq.Poller()
        poller.register(fe_event, zmq.POLLIN)
        poller.register(be_event, zmq.POLLIN)
        poller.register(be_stream, zmq.POLLIN)

        fe_event.send(b'REGISTER')
        be_event.send(fe_event.recv())

        while True:
            try:
                poll_socks = dict(poller.poll(None))
            except zmq.ZMQError:
                break  # interrupted

            if poll_socks.get(fe_event) == zmq.POLLIN:
                be_event.send_multipart(fe_event.recv_multipart())
            if poll_socks.get(be_event) == zmq.POLLIN:
                msg = be_event.recv_multipart()
                if msg[0] == 'QUIT':
                    break
                fe_event.send_multipart(msg)
            if poll_socks.get(be_stream) == zmq.POLLIN:
                fe_stream.send_multipart(be_stream.recv_multipart())


class Node(object):
    def __init__(self):
        self.nodeid = ''
        self.running = True
        ctx = zmq.Context.instance()
        self.event_io = ctx.socket(zmq.PAIR)
        self.stream_out = ctx.socket(zmq.PAIR)
        self.poller = zmq.Poller()
        self.iothread = IOThread()

    def init(self):
        ''' Final initialization. '''
        # Initialization of sockets.
        self.event_io.bind('inproc://event')
        self.stream_out.bind('inproc://stream')
        self.poller.register(self.event_io, zmq.POLLIN)

        # Start the I/O thread, and receive from it this node's ID
        self.iothread.start()
        self.nodeid = self.event_io.recv()
        print('Node started, id={}'.format(self.nodeid))

    def event(self, data, sender_id):
        ''' Event data handler. Reimplemented in Simulation. '''
        print('Received data from {}'.format(sender_id))

    def step(self):
        ''' Perform one iteration step. Reimplemented in Simulation. '''
        pass

    def start(self):
        ''' Starting of main loop. '''
        # Final Initialization
        self.init()

        # run() implements the main loop
        self.run()

        # Send quit event to the worker thread and wait for it to close.
        self.event_io.send('QUIT')
        self.iothread.join()

    def quit(self):
        ''' Quit the simulation process. '''
        self.running = False

    def run(self):
        ''' Start the main loop of this node. '''
        while self.running:
            # Get new events from the I/O thread
            while self.poll():
                sender_id = self.event_io.recv()
                data      = self.event_io.recv_pyobj()
                self.event(data, sender_id)
            # Perform a simulation step
            self.step()

            # Process timers
            Timer.update_timers()

    def poll(self):
        ''' Poll for incoming data from I/O thread '''
        try:
            poll_socks = dict(self.poller.poll(0))
            return poll_socks.get(self.event_io) == zmq.POLLIN
        except zmq.ZMQError:
            return False

    def addnodes(self, count=1):
        self.event_io.send(bytearray((count,)), zmq.SNDMORE)
        self.event_io.send(b'ADDNODES')

    def send_event(self, data, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        self.event_io.send(bytearray(stack.sender() or '*', 'ascii'), zmq.SNDMORE)
        self.event_io.send_pyobj(data)

    def send_stream(self, data, name):
        # self.stream_out.send(bytearray(name + self.nodeid, 'ascii'), zmq.SNDMORE)
        self.stream_out.send(bytearray(name, 'ascii') + self.nodeid, zmq.SNDMORE)
        self.stream_out.send_pyobj(data)
