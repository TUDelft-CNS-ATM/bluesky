""" Node encapsulates the sim process, and manages process I/O. """
import os
import zmq
import msgpack
import bluesky as bs
from bluesky import stack
from bluesky.core.walltime import Timer
from bluesky.network.npcodec import encode_ndarray, decode_ndarray


class Node:
    def __init__(self, event_port, stream_port):
        self.node_id = b'\x00' + os.urandom(4)
        self.host_id = b''
        self.running = True
        ctx = zmq.Context.instance()
        self.event_io = ctx.socket(zmq.DEALER)
        self.stream_out = ctx.socket(zmq.PUB)
        self.event_port = event_port
        self.stream_port = stream_port

    def update(self):
        ''' Update timers and perform I/O. '''
        # Process timers
        Timer.update_timers()
        # Get new events from the I/O thread
        # while self.event_io.poll(0):
        if self.event_io.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            msg = self.event_io.recv_multipart()
            route, eventname, data = msg[:-2], msg[-2], msg[-1]
            # route back to sender is acquired by reversing the incoming route
            route.reverse()
            if eventname == b'QUIT':
                self.quit()
            else:
                pydata = msgpack.unpackb(
                    data, object_hook=decode_ndarray, raw=False)
                bs.sim.event(eventname, pydata, route)

    def connect(self):
        ''' Connect node to the BlueSky server. '''
        # Initialization of sockets.
        self.event_io.setsockopt(zmq.IDENTITY, self.node_id)
        self.event_io.connect(f'tcp://localhost:{self.event_port}')
        self.stream_out.connect(f'tcp://localhost:{self.stream_port}')

        # Start communication, and receive this node's ID
        self.send_event(b'REGISTER')
        self.host_id = self.event_io.recv_multipart()[0]
        # print(f'Node connected, id={self.node_id}')

    def quit(self):
        ''' Quit the simulation process. '''
        self.running = False

    def stop(self):
        ''' Stack stop/quit command. '''
        # On a stack quit command, send quit signal to server to stop all
        # simulations.
        self.send_event(b'QUIT')

    def run(self):
        ''' Start the main loop of this node. '''
        while self.running:
            # Perform a simulation step
            self.update()
            bs.sim.update()
            # Update screen logic
            bs.scr.update()

    def addnodes(self, count=1):
        self.send_event(b'ADDNODES', count)

    def send_event(self, eventname, data=None, target=None):
        # On the sim side, target is obtained from the currently-parsed stack command
        target = target or stack.routetosender() or [b'*']
        pydata = msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
        self.event_io.send_multipart(target + [eventname, pydata])

    def send_stream(self, name, data):
        self.stream_out.send_multipart([name + self.node_id, msgpack.packb(data, default=encode_ndarray, use_bin_type=True)])
