# A Simple UDP class
import socket
import msgpack
from bluesky.network.common import get_ownip
from bluesky import settings

settings.set_variable_defaults(discovery_port=11000)

IS_SERVER = 0
IS_CLIENT = 1
IS_REQUEST = 2
IS_REPLY = 4

class Discovery:
    """simple UDP ping class"""
    handle = None   # Socket for send/recv
    port = 0        # UDP port we work on
    address = ''    # Own address
    broadcast = ''  # Broadcast address
    def __init__(self, own_id, is_client=True):
        self.address = get_ownip()
        self.broadcast = '255.255.255.255'
        self.port = settings.discovery_port
        self.own_id = own_id
        self.mask = IS_CLIENT if is_client else IS_SERVER

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

    def send(self, buf):
        self.handle.sendto(buf, 0, (self.broadcast, self.port))

    def recv(self, n):
        return self.handle.recvfrom(n)

    def send_request(self):
        data = msgpack.packb([self.mask|IS_REQUEST])
        self.send(self.own_id + data)

    def send_reply(self, eport, sport):
        data = msgpack.packb([self.mask|IS_REPLY, eport, sport])
        self.send(self.own_id + data)

    def recv_reqreply(self):
        msg, addr = self.recv(13) # The longest message is 13 bytes
        class DiscoveryReply:
            def __init__(self, msg, addr):
                self.conn_ip = addr[0]
                self.conn_id = msg[:5]
                data = msgpack.unpackb(msg[5:])
                self.is_client = data[0] & IS_CLIENT
                self.is_server = not self.is_client
                self.is_reply = data[0] & IS_REPLY
                self.is_request = not self.is_reply
                self.ports = data[1:]

            def __repr__(self):
                return 'Discovery {} received from {} {}'.format(
                    'request' if self.is_request else 'reply',
                    'client' if self.is_client else 'server',
                    self.conn_ip)

        return DiscoveryReply(msg, addr)
