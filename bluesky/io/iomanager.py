import os
import sys
from threading import Thread
from subprocess import Popen
import zmq


class IOManager(Thread):
    def __init__(self):
        super(IOManager, self).__init__()
        self.localnodes = list()
        self.running = True

    def addnodes(self, count=1):
        for _ in range(count):
            p = Popen([sys.executable, 'BlueSky_qtgl.py', '--node'])
            self.localnodes.append(p)

    def run(self):
        # Get ZMQ context
        ctx = zmq.Context.instance()

        # Convenience class for event connection handling
        class EventConn:
            # Generate one host ID for this host
            host_id = b'\x00' + os.urandom(5)

            def __init__(self, endpoint):
                self.sock = ctx.socket(zmq.ROUTER)
                self.sock.bind(endpoint)
                self.namefromid = dict()
                self.idfromname = dict()
                self.conn_count = 0

            def __eq__(self, sock):
                # Compare own socket to socket returned from poller.poll
                return self.sock == sock

            def register(self, connid):
                # The connection ID consists of the host id plus the index of the
                # new connection encoded in two bytes.
                self.conn_count += 1
                name = self.host_id + bytearray((self.conn_count // 256, self.conn_count % 256))
                self.namefromid[connid] = name
                self.idfromname[name] = connid
                return name


        # Create connection points for clients
        fe_event  = EventConn('tcp://*:9000')
        fe_stream = ctx.socket(zmq.XPUB)
        fe_stream.bind('tcp://*:9001')


        # Create connection points for sim workers
        be_event  = EventConn('tcp://*:10000')
        be_stream = ctx.socket(zmq.XSUB)
        be_stream.bind('tcp://*:10001')

        # Create poller for both event connection points and the stream reader
        poller = zmq.Poller()
        poller.register(fe_event.sock, zmq.POLLIN)
        poller.register(be_event.sock, zmq.POLLIN)
        poller.register(be_stream, zmq.POLLIN)
        poller.register(fe_stream, zmq.POLLIN)

        # Start the first simulation node
        self.addnodes()

        while self.running:
            try:
                events = dict(poller.poll(None))
            except zmq.ZMQError:
                print('ERROR while polling')
                break  # interrupted

            # The socket with incoming data
            for sock, event in events.items():
                if event != zmq.POLLIN:
                    # The event does not refer to incoming data: skip for now
                    continue
                # Receive the message
                msg = sock.recv_multipart()
                # First check if this is a stream message: these should be forwarded unprocessed.
                if sock == be_stream:
                    fe_stream.send_multipart(msg)
                elif sock == fe_stream:
                    be_stream.send_multipart(msg)
                else:
                    # Select the correct source and destination
                    src, dest = (fe_event, be_event) if sock == fe_event else (be_event, fe_event)

                    if msg[-1] == b'REGISTER':
                        # This is a registration message for a new connection
                        # Send reply with connection name
                        sock.send_multipart([msg[0], src.register(msg[0])])

                    elif msg[-1] == b'ADDNODES':
                        # This is a request to start new nodes.
                        count = msg[-2]
                        self.addnodes(count)

                    elif msg[-1] == b'QUIT':
                        # TODO: send quit to all
                        self.running = False

                    else:
                        # This is a regular message that should be forwarded
                        # Swap sender and target so that msg is sent to target
                        sender = src.namefromid.get(msg[0]) or b'unknown'
                        target = dest.idfromname.get(msg[1]) or b'*'
                        msg[:2] = target, sender
                        if target == b'*':
                            # This is a send-to-all message
                            for connid in dest.namefromid:
                                msg[0] = connid
                                dest.sock.send_multipart(msg)
                        else:
                            dest.sock.send_multipart(msg)

        # Wait for all nodes to finish
        for n in self.localnodes:
            n.wait()
