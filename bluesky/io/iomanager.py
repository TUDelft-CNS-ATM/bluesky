''' BlueSky I/O manager. '''
import os
from multiprocessing import cpu_count
import sys
from threading import Thread
from subprocess import Popen
import zmq
import msgpack

# Local imports
from bluesky import settings


# Register settings defaults
settings.set_variable_defaults(max_nnodes=cpu_count())

def split_scenarios(scentime, scencmd):
    ''' Split the contents of a batch file into individual scenarios. '''
    start = 0
    for i in range(1, len(scencmd) + 1):
        if i == len(scencmd) or scencmd[i][:4] == 'SCEN':
            scenname = scencmd[start].split()[1].strip()
            yield (scenname, scentime[start:i], scencmd[start:i])
            start = i


class IOManager(Thread):
    ''' Implementation of the BlueSky I/O manager server. '''
    def __init__(self):
        super(IOManager, self).__init__()
        self.spawned_processes = list()
        self.running           = True
        self.max_nnodes        = min(cpu_count(), settings.max_nnodes)
        self.scenarios         = []
        self.host_id           = b'\x00' + os.urandom(4)
        self.clients           = []
        self.workers           = []
        self.servers           = {self.host_id : dict(route=[], nodes=self.workers)}

    def addnodes(self, count=1):
        ''' Add [count] nodes to this server. '''
        for _ in range(count):
            p = Popen([sys.executable, 'BlueSky_qtgl.py', '--node'])
            self.spawned_processes.append(p)

    def run(self):
        ''' The main loop of this server. '''
        # Get ZMQ context
        ctx = zmq.Context.instance()

        # Create connection points for clients
        fe_event  = ctx.socket(zmq.ROUTER)
        fe_event.setsockopt(zmq.IDENTITY, self.host_id)
        fe_event.bind('tcp://*:9000')
        fe_stream = ctx.socket(zmq.XPUB)
        fe_stream.bind('tcp://*:9001')


        # Create connection points for sim workers
        be_event  = ctx.socket(zmq.ROUTER)
        be_event.setsockopt(zmq.IDENTITY, self.host_id)
        be_event.bind('tcp://*:10000')
        be_stream = ctx.socket(zmq.XSUB)
        be_stream.bind('tcp://*:10001')

        # Create poller for both event connection points and the stream reader
        poller = zmq.Poller()
        poller.register(fe_event, zmq.POLLIN)
        poller.register(be_event, zmq.POLLIN)
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

                    # Message format: [route0, ..., routen, name, data]
                    route, eventname, data = msg[:-2], msg[-2], msg[-1]
                    sender_id = route[0]

                    if eventname == b'REGISTER':
                        # This is a registration message for a new connection
                        # Reply with our host ID
                        src.send_multipart([sender_id, self.host_id, b'REGISTER', b''])
                        # Notify clients of this change
                        if srcisclient:
                            self.clients.append(sender_id)
                            # If the new connection is a client, send it our server list
                            src.send_multipart(route + [b'NODESCHANGED', msgpack.packb(self.servers, use_bin_type=True)])
                        else:
                            self.workers.append(sender_id)
                            data = msgpack.packb({self.host_id : self.servers[self.host_id]}, use_bin_type=True)
                            for client_id in self.clients:
                                dest.send_multipart([client_id, self.host_id, b'NODESCHANGED', data])
                        continue # No message needs to be forwarded

                    elif eventname == b'NODESCHANGED':
                        servers_upd = msgpack.unpackb(data, encoding='utf-8')
                        # Update the route with a hop to the originating server
                        for server in servers_upd.values():
                            server['route'].insert(0, sender_id)
                        self.servers.update(servers_upd)
                        # Notify own clients of this change
                        data = msgpack.packb(servers_upd, use_bin_type=True)
                        for client_id in self.clients:
                            # Skip sender to avoid infinite message loop
                            if client_id != sender_id:
                                fe_event.send_multipart([client_id, self.host_id, b'NODESCHANGED', data])

                    elif eventname == b'ADDNODES':
                        # This is a request to start new nodes.
                        count = msgpack.unpackb(data)
                        self.addnodes(count)
                        continue # No message needs to be forwarded

                    elif eventname == b'STATECHANGE':
                        state = msgpack.unpackb(data)
                        continue

                    elif eventname == b'QUIT':
                        # Send quit to all nodes
                        target_id = b'*'
                        self.running = False

                    elif eventname == b'BATCH':
                        scentime, scencmd = msgpack.unpackb(data, encoding='utf-8')
                        self.scenarios = [scen for scen in split_scenarios(scentime, scencmd)]
                        # Check if the batch list contains scenarios
                        if not self.scenarios:
                            echomsg = 'No scenarios defined in batch file!'
                        else:
                            echomsg = 'Found {} scenarios in batch'.format(len(self.scenarios))
                            # # Available nodes (nodes that are in init or hold mode):
                            # av_nodes = [n for n, conn in enumerate(self.connections) if conn[2] in [0, 2]]
                            # for i in range(min(len(av_nodes), len(self.scenarios))):
                            #     self.sendScenario(self.connections[i])
                            # # If there are still scenarios left, determine and start the required number of local nodes
                            # reqd_nnodes = min(len(self.scenarios), max(0, self.max_nnodes - len(self.localnodes)))
                            # for n in range(reqd_nnodes):
                            #     self.addNode()
                        # ECHO the results to the calling client
                        eventname = b'ECHO'
                        data = msgpack.packb(echomsg, use_bin_type=True)

                    # ============================================================
                    # If we get here there is a message that needs to be forwarded
                    # Cycle the route by one step to get the next hop in the route
                    # (or the destination)
                    route.append(route.pop(0))
                    msg    = route + [eventname, data]
                    if route[0] == b'*':
                        # This is a send-to-all message
                        for connid in self.clients if srcisclient else self.workers:
                            msg[0] = connid
                            dest.send_multipart(msg)
                    else:
                        dest.send_multipart(msg)

        # Wait for all nodes to finish
        for n in self.spawned_processes:
            n.wait()
