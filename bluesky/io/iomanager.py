import os
import sys
from threading import Thread
from subprocess import Popen
import zmq
import msgpack


class IOManager(Thread):
    def __init__(self):
        super(IOManager, self).__init__()
        self.spawned_processes = list()
        self.running = True
        self.nodes = dict()

    def addnodes(self, count=1):
        for _ in range(count):
            p = Popen([sys.executable, 'BlueSky_qtgl.py', '--node'])
            self.spawned_processes.append(p)

    def run(self):
        # Get ZMQ context
        ctx = zmq.Context.instance()

        class EventConn:
            ''' Convenience class for event connection handling. '''
            # Generate one host ID for this host
            host_id = b'\x00' + os.urandom(4)

            def __init__(self, endpoint, connection_key):
                self.sock = ctx.socket(zmq.ROUTER)
                self.sock.bind(endpoint)
                self.namefromid = dict()
                self.idfromname = dict()
                self.conn_count = 0
                self.conn_key   = connection_key

            def __eq__(self, sock):
                # Compare own socket to socket returned from poller.poll
                return self.sock == sock

            def register(self, connid):
                # The connection ID consists of the host id plus the index of the
                # new connection encoded in two bytes.
                self.conn_count += 1
                name = self.host_id + self.conn_key + \
                    bytearray((self.conn_count // 256, self.conn_count % 256))
                self.namefromid[connid] = name
                self.idfromname[name] = connid
                return name


        # Create connection points for clients
        fe_event  = EventConn('tcp://*:9000', b'c')
        fe_stream = ctx.socket(zmq.XPUB)
        fe_stream.bind('tcp://*:9001')


        # Create connection points for sim workers
        be_event  = EventConn('tcp://*:10000', b'w')
        be_stream = ctx.socket(zmq.XSUB)
        be_stream.bind('tcp://*:10001')

        # We start with zero nodes
        self.nodes[EventConn.host_id] = []

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
                    srcisclient = (sock == fe_event)
                    src, dest = (fe_event, be_event) if srcisclient else (be_event, fe_event)

                    # Message format: [sender, target, name, data]
                    sender, target, eventname, data = msg

                    if eventname == b'REGISTER':
                        # This is a registration message for a new connection
                        # Send reply with connection name
                        newid = src.register(sender)
                        msg[-1] = newid
                        src.sock.send_multipart(msg)
                        # Notify clients of this change
                        if srcisclient:
                            src.sock.send_multipart([sender, src.host_id, b'NODESCHANGED', msgpack.packb(self.nodes, use_bin_type=True)])
                        else:
                            self.nodes[src.host_id].append(newid)
                            data = msgpack.packb({src.host_id : self.nodes[src.host_id]}, use_bin_type=True)
                            for connid in dest.namefromid:
                                dest.sock.send_multipart([connid, src.host_id, b'NODESCHANGED', data])

                    elif eventname == b'ADDNODES' and target == fe_event.host_id:
                        # This is a request to start new nodes.
                        count = msgpack.unpackb(data)
                        self.addnodes(count)

                    elif eventname == b'QUIT':
                        # Send quit to all nodes
                        for connid in be_event.namefromid:
                            be_event.sock.send_multipart([connid, be_event.host_id, b'QUIT', b''])
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
        for n in self.spawned_processes:
            n.wait()
