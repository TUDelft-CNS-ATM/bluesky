# A Simple UDP class
import socket
import msgpack
import zmq
from bluesky.core.signal import Signal
from bluesky.network.common import get_ownip, IDLEN
from bluesky import settings

settings.set_variable_defaults(discovery_port=11000)

IS_SERVER = 0
IS_CLIENT = 1
IS_REQUEST = 2
IS_REPLY = 4

class Discovery:
    """simple UDP ping class"""
    def __init__(self, own_id, is_client=True):
        self.address = get_ownip()
        self.broadcast = '255.255.255.255'
        self.port = settings.discovery_port
        self.own_id = own_id
        self.mask = IS_CLIENT if is_client else IS_SERVER

        # When run as a client, typically our own poller is used
        # In this case, also a signal is emitted when one or more
        # servers are discovered
        self.poller = None
        self.server_discovered = Signal('server-discovered')

        # Create UDP socket
        self.handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # Ask operating system to let us do broadcasts from socket
        self.handle.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            self.handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        else:
            self.handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind UDP socket to local port so we can receive pings
        self.handle.bind(('', self.port))

    def start(self):
        ''' Start UDP-based discovery of available BlueSky servers. '''
        if not self.poller:
            self.poller = zmq.Poller()
            self.poller.register(self.handle, zmq.POLLIN)
            self.send_request()

    def stop(self):
        ''' Stop UDP-based discovery. '''
        if self.poller:
            self.poller.unregister(self.handle)
            self.poller = None

    def update(self):
        ''' Periodically call this function to find remote servers. 
            The 'server-discovered' Signal is emitted whenever a
            server is found.
        '''
        if self.poller is None:
            return
        events = dict(self.poller.poll(0))

        # The socket with incoming data
        for sock, event in events.items():
            if event != zmq.POLLIN:
                # The event does not refer to incoming data: skip for now
                continue
            # Get the incoming data
            dmsg = self.recv_reqreply()
            if dmsg.conn_id != self.own_id and dmsg.is_server:
                self.server_discovered.emit(dmsg.conn_ip, dmsg.ports)

        self.send_request()


    def send(self, buf):
        self.handle.sendto(buf, 0, (self.broadcast, self.port))

    def receive(self, n):
        return self.handle.recvfrom(n)

    def send_request(self):
        data = msgpack.packb([self.mask|IS_REQUEST])
        self.send(self.own_id + data)

    def send_reply(self, eport, sport):
        data = msgpack.packb([self.mask|IS_REPLY, eport, sport])
        self.send(self.own_id + data)

    def recv_reqreply(self):
        msg, addr = self.receive(IDLEN + 8) # The longest message is IDLEN + 8 bytes
        class DiscoveryReply:
            def __init__(self, msg, addr):
                self.conn_ip = addr[0]
                self.conn_id = msg[:IDLEN]
                data = msgpack.unpackb(msg[IDLEN:])
                self.is_client = bool(data[0] & IS_CLIENT)
                self.is_server = not self.is_client
                self.is_reply = bool(data[0] & IS_REPLY)
                self.is_request = not self.is_reply
                self.ports = data[1:]

            def __repr__(self):
                return 'Discovery {} received from {} {}'.format(
                    'request' if self.is_request else 'reply',
                    'client' if self.is_client else 'server',
                    self.conn_ip)

        return DiscoveryReply(msg, addr)
