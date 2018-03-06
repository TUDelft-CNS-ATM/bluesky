# A Simple UDP class
import socket
from bluesky.network.common import get_ownip


class UDP(object):
    """simple UDP ping class"""
    handle = None   # Socket for send/recv
    port = 0        # UDP port we work on
    address = ''    # Own address
    broadcast = ''  # Broadcast address

    def __init__(self, port, address=None, broadcast=None):
        self.address = address or get_ownip()
        self.broadcast = broadcast or '255.255.255.255'
        self.port = port
        # Create UDP socket
        self.handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # Ask operating system to let us do broadcasts from socket
        self.handle.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        # Bind UDP socket to local port so we can receive pings
        self.handle.bind(('', port))

    def send(self, buf):
        self.handle.sendto(buf, 0, (self.broadcast, self.port))

    def recv(self, n):
        return self.handle.recvfrom(n)
