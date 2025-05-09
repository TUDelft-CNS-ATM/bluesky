''' Server test '''
import sys
import signal
from subprocess import Popen
from multiprocessing import cpu_count
from threading import Thread

import zmq
import msgpack

import bluesky as bs
from bluesky.network.npcodec import encode_ndarray
from bluesky.network.discovery import Discovery
from bluesky.network.common import genid, bin2hex, MSG_SUBSCRIBE, MSG_UNSUBSCRIBE, GROUPID_SIM, IDLEN


# Register settings defaults
bs.settings.set_variable_defaults(max_nnodes=cpu_count(),
                                  recv_port=11000, send_port=11001,
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
    def __init__(self, discovery, altconfig=None, startscn=None):
        super().__init__()
        self.spawned_processes = dict()
        self.running = True
        self.max_nnodes = min(cpu_count(), bs.settings.max_nnodes)
        self.scenarios = []
        self.server_id = genid(groupid=GROUPID_SIM, seqidx=0)
        self.max_group_idx = 0
        self.sim_nodes = set()
        self.all_nodes = set()
        self.avail_nodes = set()
        
        # Information to pass on to spawned nodes
        self.altconfig = altconfig
        self.startscn = startscn

        if bs.settings.enable_discovery or discovery:
            self.discovery = Discovery(self.server_id, is_client=False)
        else:
            self.discovery = None

        # Get ZMQ context
        ctx = zmq.Context.instance()
        self.sock_recv = ctx.socket(zmq.XSUB)
        self.sock_send = ctx.socket(zmq.XPUB)
        self.poller = zmq.Poller()

        # Connect to interrupt signal
        signal.signal(signal.SIGINT, lambda *args: self.quit())
        signal.signal(signal.SIGTERM, lambda *args: self.quit())

    def quit(self):
        self.running = False

    def sendscenario(self, node_id):
        # Send a new scenario to the target sim process
        scen = self.scenarios.pop(0)
        self.send(b'BATCH', scen, node_id)

    def addnodes(self, count=1, node_ids=None, startscn=None):
        ''' Add [count] nodes to this server. '''
        for idx in range(count):
            if node_ids:
                newid = genid(node_ids[idx])
            else:
                self.max_group_idx += 1
                newid = genid(self.server_id[:-1], seqidx=self.max_group_idx)
            args = [sys.executable, '-m', 'bluesky', '--sim', '--groupid', bin2hex(newid)]
            if self.altconfig:
                args.extend(['--configfile', self.altconfig])
            if startscn:
                args.extend(['--scenfile', startscn])
            self.spawned_processes[newid] = Popen(args)

    def send(self, topic, data='', dest=b''):
        self.sock_send.send_multipart(
            [
                dest.ljust(IDLEN, b'*') + topic + self.server_id,
                msgpack.packb(data, default=encode_ndarray, use_bin_type=True)
            ]
        )

    def run(self):
        ''' The main loop of this server. '''
        print(f'Starting server with id', self.server_id)
        # For the server, send/recv ports are reversed
        self.sock_recv.bind(f'tcp://*:{bs.settings.send_port}')
        self.sock_send.bind(f'tcp://*:{bs.settings.recv_port}')


        # Create poller for pub/sub and discovery
        self.poller.register(self.sock_recv, zmq.POLLIN)
        self.poller.register(self.sock_send, zmq.POLLIN)

        if self.discovery:
            self.poller.register(self.discovery.handle, zmq.POLLIN)
        print(f'Discovery is {"en" if self.discovery else "dis"}abled')

        # Create subscription for messages targeted at this server
        self.sock_recv.send_multipart([b'\x01' + self.server_id])

        # Start the first simulation node
        self.addnodes(startscn=self.startscn)

        while self.running:
            try:
                events = dict(self.poller.poll(None))
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
                    if dmsg.conn_id != self.server_id and dmsg.is_request:
                        # This is a request from someone else: send a reply
                        self.discovery.send_reply(bs.settings.recv_port,
                            bs.settings.send_port)
                    continue

                # Receive the message
                msg = sock.recv_multipart()
                if not msg:
                    # In the rare case that a message is empty, skip remaning processing
                    continue
                if sock == self.sock_send:
                    # This is an (un)subscribe message. If it's an id-only subscription
                    # this is also a registration message
                    if len(msg[0]) == IDLEN + 1:
                        if msg[0][0] == MSG_SUBSCRIBE:
                            # This is an initial client, server, or node subscription
                            if msg[0][1] == GROUPID_SIM and msg[0][1:] in self.spawned_processes:
                                # This is a node owned by this server which has successfully started.
                                self.sim_nodes.add(msg[0][1:])
                                self.send(b'REQUEST', ['STATECHANGE'], msg[0][1:])
                        elif msg[0][0] == MSG_UNSUBSCRIBE and msg[0][1] == GROUPID_SIM and msg[0][1:] in self.spawned_processes:
                            print('Removing node', msg[0][1:])
                            self.sim_nodes.discard(msg[0][1:])
                    # Always forward
                    self.sock_recv.send_multipart(msg)
                elif sock == self.sock_recv:
                    # First check if message is directed at this server
                    if msg[0].startswith(self.server_id):
                        topic, sender_id = msg[0][IDLEN:-IDLEN], msg[0][-IDLEN:]
                        data = msgpack.unpackb(msg[1], raw=False)
                        # TODO: also use Signal logic in server?
                        if topic == b'QUIT':
                            self.quit()
                        elif topic == b'ADDNODES':
                            if isinstance(data, int):
                                self.addnodes(count=data)
                            elif isinstance(data, dict):
                                self.addnodes(**data)
                        elif topic == b'STATECHANGE':
                            state = data[1]['simstate']
                            if state < bs.OP:
                                # If we have batch scenarios waiting, send
                                # the simulation node a new scenario, otherwise store it in
                                # the available simulation node list
                                if self.scenarios:
                                    self.sendscenario(sender_id)
                                else:
                                    self.avail_nodes.add(sender_id)
                            else:
                                self.avail_nodes.discard(sender_id)
                        elif topic == b'BATCH':
                            scentime, scencmd = data
                            self.scenarios = [scen for scen in split_scenarios(scentime, scencmd)]
                            # Check if the batch list contains scenarios
                            if not self.scenarios:
                                echomsg = 'No scenarios defined in batch file!'
                            else:
                                echomsg = f'Found {len(self.scenarios)} scenarios in batch'
                                # Send scenario to available nodes (nodes that are in init or hold mode):
                                while self.avail_nodes and self.scenarios:
                                    node_id = next(iter(self.avail_nodes))
                                    self.sendscenario(node_id)
                                    self.avail_nodes.discard(node_id)

                                # If there are still scenarios left, determine and
                                # start the required number of local nodes
                                reqd_nnodes = min(len(self.scenarios), max(0, self.max_nnodes - len(self.sim_nodes)))
                                self.addnodes(reqd_nnodes)
                            # ECHO the results to the calling client
                            topic = b'ECHO'
                            data = msgpack.packb(dict(text=echomsg, flags=0), use_bin_type=True)
                            self.sock_send.send_multipart([sender_id + topic + self.server_id, data])
                    else:
                        self.sock_send.send_multipart(msg)
        print('Server quit. Stopping nodes:')
        for pid, p in self.spawned_processes.items():
            # print('Stopping node:', pid, end=' ')
            # p.send_signal(signal.SIGTERM)
            p.terminate()
            p.wait()
            # Inform network that node is removed
            self.sock_recv.send_multipart([b'\x00' + pid])
            # print('done')
        print('Closing connections:', end=' ')
        self.poller.unregister(self.sock_recv)
        self.poller.unregister(self.sock_send)
        self.sock_recv.close()
        self.sock_send.close()
        zmq.Context.instance().destroy()
        print('done')
