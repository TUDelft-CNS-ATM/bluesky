""" Node encapsulates the sim process, and manages process I/O. """
from threading import Thread
import zmq
import bluesky as bs
from bluesky import stack
from .timer import Timer


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
        fe_event.send('REGISTER')
        be_event.send(fe_event.recv())

        be_event.connect('inproc://event')
        be_stream.connect('inproc://stream')
        poller = zmq.Poller()
        poller.register(fe_event, zmq.POLLIN)
        poller.register(be_event, zmq.POLLIN)
        poller.register(be_stream, zmq.POLLIN)

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
        ctx = zmq.Context.instance()
        self.event = ctx.socket(zmq.PAIR)
        self.stream = ctx.socket(zmq.PAIR)
        self.poller = zmq.Poller()

    def start(self):
        ''' Final initialization and starting of main loop. '''
        # Initialization of sockets.
        self.event.bind('inproc://event')
        self.stream.bind('inproc://stream')
        self.poller.register(self.event, zmq.POLLIN)

        # Start the I/O thread, and receive from it this node's ID
        iothread = IOThread()
        iothread.start()
        self.nodeid = self.event.recv()

        bs.sim.prepare()
        self.run()

        # Send quit event to the worker thread and wait for it to close.
        self.event.send('QUIT')
        iothread.join()

    def run(self):
        ''' Start the main loop of this node. '''
        while bs.sim.running:
            # Get new events from the I/O thread
            while self.poll():
                sender_id = self.event.recv()
                data      = self.event.recv_pyobj()
                bs.sim.event(data, sender_id)
            # Perform a simulation step
            bs.sim.step()

            # Process timers
            Timer.update_timers()

    def poll(self):
        ''' Poll for incoming data from I/O thread '''
        try:
            poll_socks = dict(self.poller.poll(0))
            return poll_socks.get(self.event) == zmq.POLLIN
        except zmq.ZMQError:
            return None

    def send_event(self, data, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        self.event.send(stack.sender() or '*', zmq.SNDMORE)
        self.event.send_pyobj(data)

    def send_stream(self, data, name):
        self.stream.send(name + self.nodeid, zmq.SNDMORE)
        self.stream.send_pyobj(data)
