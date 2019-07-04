''' BlueSky simulation server. '''
import os
from multiprocessing import cpu_count
from threading import Thread
import sys
from subprocess import Popen
import zmq
import msgpack

# Local imports
import bluesky as bs

from .discovery import Discovery


# Register settings defaults
bs.settings.set_variable_defaults(max_nnodes=cpu_count(),
                                  event_port=9000, stream_port=9001,
                                  simevent_port=10000, simstream_port=10001,
                                  enable_discovery=False)

def split_scenarios(scentime, scencmd):
    ''' Split the contents of a batch file into individual scenarios. '''
    start = 0
    for i in range(1, len(scencmd) + 1):
        if i == len(scencmd) or scencmd[i][:4] == 'SCEN':
            scenname = scencmd[start].split()[1].strip()
            yield dict(name=scenname, scentime=scentime[start:i], scencmd=scencmd[start:i])
            start = i


class Server(Thread):
    ''' Implementation of the BlueSky simulation server. '''

    def __init__(self, headless):
        super(Server, self).__init__()
        self.spawned_processes = list()
        self.running = True
        self.max_nnodes = min(cpu_count(), bs.settings.max_nnodes)
        self.scenarios = []
        self.host_id = b'\x00' + os.urandom(4)
        self.clients = []
        self.workers = []
        self.servers = {self.host_id : dict(route=[], nodes=self.workers)}
        self.avail_workers = dict()

        if bs.settings.enable_discovery or headless:
            self.discovery = Discovery(self.host_id, is_client=False)
        else:
            self.discovery = None

    def sendScenario(self, worker_id):
        # Send a new scenario to the target sim process
        scen = self.scenarios.pop(0)
        data = msgpack.packb(scen)
        self.be_event.send_multipart([worker_id, self.host_id, b'BATCH', data])

    def addnodes(self, count=1):
        ''' Add [count] nodes to this server. '''
        for _ in range(count):
            p = Popen([sys.executable, 'BlueSky.py', '--sim'])
            self.spawned_processes.append(p)

    def run(self):
        ''' The main loop of this server. '''
        # Get ZMQ context
        ctx = zmq.Context.instance()

        # Create connection points for clients
        self.fe_event = ctx.socket(zmq.ROUTER)
        self.fe_event.setsockopt(zmq.IDENTITY, self.host_id)
        self.fe_event.bind('tcp://*:{}'.format(bs.settings.event_port))
        self.fe_stream = ctx.socket(zmq.XPUB)
        self.fe_stream.bind('tcp://*:{}'.format(bs.settings.stream_port))
        print('Accepting event connections on port {}, and stream connections on port {}'.format(
            bs.settings.event_port, bs.settings.stream_port))

        # Create connection points for sim workers
        self.be_event  = ctx.socket(zmq.ROUTER)
        self.be_event.setsockopt(zmq.IDENTITY, self.host_id)
        self.be_event.bind('tcp://*:{}'.format(bs.settings.simevent_port))
        self.be_stream = ctx.socket(zmq.XSUB)
        self.be_stream.bind('tcp://*:{}'.format(bs.settings.simstream_port))

        # Create poller for both event connection points and the stream reader
        poller = zmq.Poller()
        poller.register(self.fe_event, zmq.POLLIN)
        poller.register(self.be_event, zmq.POLLIN)
        poller.register(self.be_stream, zmq.POLLIN)
        poller.register(self.fe_stream, zmq.POLLIN)

        if self.discovery:
            poller.register(self.discovery.handle, zmq.POLLIN)
        print('Discovery is {}abled'.format('en' if self.discovery else 'dis'))

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

                # First check if the poller was triggered by the discovery socket
                if self.discovery and sock == self.discovery.handle.fileno():
                    # This is a discovery message
                    dmsg = self.discovery.recv_reqreply()
                    # print('Received', dmsg)
                    if dmsg.conn_id != self.host_id and dmsg.is_request:
                        # This is a request from someone else: send a reply
                        # print('Sending reply')
                        self.discovery.send_reply(bs.settings.event_port,
                            bs.settings.stream_port)
                    continue
                # Receive the message
                msg = sock.recv_multipart()

                # Check if this is a stream message: these should be forwarded unprocessed.
                if sock == self.be_stream:
                    self.fe_stream.send_multipart(msg)
                elif sock == self.fe_stream:
                    self.be_stream.send_multipart(msg)
                else:
                    # Select the correct source and destination
                    srcisclient = (sock == self.fe_event)
                    src, dest = (self.fe_event, self.be_event) if srcisclient else (self.be_event, self.fe_event)

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
                            data = msgpack.packb(self.servers, use_bin_type=True)
                            src.send_multipart([sender_id, self.host_id, b'NODESCHANGED', data])
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
                                self.fe_event.send_multipart([client_id, self.host_id, b'NODESCHANGED', data])

                    elif eventname == b'ADDNODES':
                        # This is a request to start new nodes.
                        count = msgpack.unpackb(data)
                        self.addnodes(count)
                        continue # No message needs to be forwarded

                    elif eventname == b'STATECHANGE':
                        state = msgpack.unpackb(data)
                        if state < bs.OP:
                            # If we have batch scenarios waiting, send
                            # the worker a new scenario, otherwise store it in
                            # the available worker list
                            if self.scenarios:
                                self.sendScenario(sender_id)
                            else:
                                self.avail_workers[sender_id] = route
                        else:
                            self.avail_workers.pop(route[0], None)
                        continue

                    elif eventname == b'QUIT':
                        self.running = False
                        # Send quit to all nodes and clients
                        msg = [self.host_id, eventname, data]
                        for connid in self.workers:
                            self.be_event.send_multipart([connid] + msg)
                        for connid in self.clients:
                            self.fe_event.send_multipart([connid] + msg)
                        continue

                    elif eventname == b'BATCH':
                        scentime, scencmd = msgpack.unpackb(data, encoding='utf-8')
                        self.scenarios = [scen for scen in split_scenarios(scentime, scencmd)]
                        # Check if the batch list contains scenarios
                        if not self.scenarios:
                            echomsg = 'No scenarios defined in batch file!'
                        else:
                            echomsg = 'Found {} scenarios in batch'.format(len(self.scenarios))
                            # Send scenario to available nodes (nodes that are in init or hold mode):
                            while self.avail_workers and self.scenarios:
                                worker_id = next(iter(self.avail_workers))
                                self.sendScenario(worker_id)
                                self.avail_workers.pop(worker_id)

                            # If there are still scenarios left, determine and
                            # start the required number of local nodes
                            reqd_nnodes = min(len(self.scenarios), max(0, self.max_nnodes - len(self.workers)))
                            self.addnodes(reqd_nnodes)
                        # ECHO the results to the calling client
                        eventname = b'ECHO'
                        data = msgpack.packb(dict(text=echomsg, flags=0), use_bin_type=True)

                    # ============================================================
                    # If we get here there is a message that needs to be forwarded
                    # Cycle the route by one step to get the next hop in the route
                    # (or the destination)
                    route.append(route.pop(0))
                    msg = route + [eventname, data]
                    if route[0] == b'*':
                        # This is a send-to-all message
                        msg.insert(0, b'')
                        for connid in self.workers if srcisclient else self.clients:
                            msg[0] = connid
                            dest.send_multipart(msg)
                    else:
                        dest.send_multipart(msg)

        # Wait for all nodes to finish
        for n in self.spawned_processes:
            n.wait()
