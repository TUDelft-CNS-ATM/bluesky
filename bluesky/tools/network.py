""" Network functionality for BlueSky."""
import time
import socket
import threading
import bluesky as bs
import sys


def as_bytes(msg):
    """
    Encodes strings to bytes.
    """
    if sys.version_info.major == 3:
        return msg.encode('utf-8')
    else:
        return msg


class TcpSocket(object):
    """A TCP Client receving message from server, analysing the data, and """
    def __init__(self):
        self.buffer_size = 1024
        self.is_connected = False
        self.receiver_thread = threading.Thread(target=self.receiver)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def connectToHost(self, ip, port):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setblocking(0)
            self.sock.settimeout(10)    # 10 second timeout
            self.sock.connect((ip, port))       # connecting
            self.is_connected = True
            print("Server connected. HOST: %s, PORT: %s" % (ip, port))
        except Exception as err:
            self.is_connected = False
            print("Connection Error: %s" % err)
            pass

    def disconnectFromHost(self):
        try:
            self.sock.close()
            self.is_connected = False
            print("Server disconnected.")
        except Exception as err:
            print("Disconnection Error: %s" % err)
            pass

    def isConnected(self):
        return self.is_connected

    def receiver(self):
        while True:
            if not self.is_connected:
                time.sleep(1.0)
                continue

            try:
                data = self.sock.recv(self.buffer_size)
                self.processData(data)
                time.sleep(0.1)
            except Exception as err:
                print("Receiver Error: %s" % err)
                time.sleep(1.0)

    def processData(self, data):
        # rewrite this function
        print("parsing data...")

    def numConnections(self):
        return None


class TcpServer(object):
    def __init__(self):
        pass

    def sendReply(self, event):
        pass

    def start(self):
        pass

    def processData(self, data, sender_id):
        pass
