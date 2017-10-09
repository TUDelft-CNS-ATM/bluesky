import os
from threading import Thread
import zmq


class Manager(Thread):
    def __init__(self):
        super(Manager, self).__init__()
        self.host_id = b'\x00' + os.urandom(4)


    def forward_event(self, src, dest, srcnames, destnames):
        msg = src.recv_multipart()
        if msg[-1] == 'REGISTER':
            # This is a registration message for a new connection
            connid = self.host_id + chr(100 + len(srcnames))
            srcnames[msg[0]] = connid
            # Send reply with connection ID
            src.send_multipart([msg[0], connid])

        else:
            # This is a message that should be forwarded
            # Swap sender and target so that msg is sent to target
            sender = srcnames.get(msg[0]) or 'unknown'
            target = msg[1]
            msg[:2] = target, sender
            if target == '*':
                # This is a send-to-all message
                for name in destnames:
                    msg[0] = name
                    dest.send_multipart(msg)
            else:
                dest.send_multipart(msg)

    def run(self):
        # Keep track of all clients and workers
        clients = dict()
        workers = dict()

        # Get ZMQ context
        ctx = zmq.Context.instance()

        # Create connection points for clients
        fe_event  = ctx.socket(zmq.ROUTER)
        fe_stream = ctx.socket(zmq.XPUB)
        fe_event.bind('tcp://*:9000')
        fe_stream.bind('tcp://*:9001')


        # Create connection points for sim workers
        be_event  = ctx.socket(zmq.ROUTER)
        be_stream = ctx.socket(zmq.XSUB)

        be_event.bind('tcp://*:10000')
        be_stream.bind('tcp://*:10001')

        # Create poller for both event connection points and the stream reader
        poller = zmq.Poller()
        poller.register(fe_event, zmq.POLLIN)
        poller.register(be_event, zmq.POLLIN)
        poller.register(be_stream, zmq.POLLIN)
        poller.register(fe_stream, zmq.POLLIN)

        while True:
            try:
                events = dict(poller.poll(None))
            except zmq.ZMQError:
                break  # interrupted

            # The socket with incoming data
            for sock in events:
                if sock == fe_event:
                    self.forward_event(fe_event, be_event, clients, workers)
                elif sock == be_event:
                    self.forward_event(be_event, fe_event, workers, clients)
                elif sock == be_stream:
                    fe_stream.send_multipart(sock.recv_multipart())
                elif sock == fe_stream:
                    be_stream.send_multipart(sock.recv_multipart())

    def send_event(self, event, target=None):
        ''' Send event placeholder. In the main process, data should be sent with
            a Client. '''
        raise RuntimeError('Data cannot be sent directly using the main process manager, use a Client instead.')


    def send_stream(self, data, name):
        ''' Send stream placeholder. In the main process, data should be sent with
            a Client. '''
        raise RuntimeError('Data cannot be sent directly using the main process manager, use a Client instead.')
