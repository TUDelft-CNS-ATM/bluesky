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


# if not 'is sim': # old variable
#     try:
#         from PyQt5.QtCore import pyqtSlot
#         from PyQt5.QtNetwork import QTcpServer, QTcpSocket
#     except ImportError:
#         from PyQt4.QtCore import pyqtSlot
#         from PyQt4.QtNetwork import QTcpServer, QTcpSocket
#
#
#     class TcpSocket(QTcpSocket):
#         def __init__(self, parent=None):
#             super(TcpSocket, self).__init__(parent)
#             self.error.connect(self.onError)
#             self.connected.connect(self.onConnected)
#             self.disconnected.connect(self.onDisconnected)
#
#         @pyqtSlot()
#         def onError(self):
#             print(self.errorString())
#
#         @pyqtSlot()
#         def onConnected(self):
#             print('TcpClient connected')
#
#         @pyqtSlot()
#         def onDisconnected(self):
#             del self.parent().connections[id(self)]
#             print('TcpClient disconnected')
#
#         def isConnected(self):
#             return (self.state() == self.ConnectedState)
#
#         @pyqtSlot()
#         def onReadyRead(self):
#             self.processData(self.readAll())
#
#         def sendReply(self, msg):
#             self.writeData(
#                 as_bytes(
#                     '{}\n'.format(msg)))
#
#         def processData(self, data):
#             # Placeholder function; override it with your own implementation
#             print('TcpSocket received', data)
#
#
#     class TcpServer(QTcpServer):
#         def __init__(self, parent=None):
#             super(TcpServer, self).__init__(parent)
#             self.connections = dict()
#
#         def incomingConnection(self, socketDescriptor):
#             newconn = TcpSocket(self)
#             newconn.setSocketDescriptor(socketDescriptor)
#             newconn.readyRead.connect(self.onReadyRead)
#             self.connections[id(newconn)] = newconn
#
#         @pyqtSlot()
#         def onReadyRead(self):
#             sender_id = id(self.sender())
#             data      = self.sender().readAll()
#             self.processData(data, sender_id)
#
#         def sendReply(self, event):
#             if event.sender_id:
#                 self.connections[event.sender_id].sendReply(event.disptext)
#
#         def processData(self, sender_id, data):
#             # Placeholder function; override it with your own implementation
#             print('TcpServer received', data, 'from sender no', sender_id)
#
#         def numConnections(self):
#             return len(self.connections.keys())
#
#
# else:
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


class StackTelnetServer(TcpServer):
    @staticmethod
    def dummy_process(cmd, sender_id):
        pass

    def __init__(self):
        super(StackTelnetServer, self).__init__()
        self.process = StackTelnetServer.dummy_process

    def connect(self, fun):
        self.process = fun

    def processData(self, data, sender_id):
        msg = bytearray(data).decode(encoding='ascii', errors='ignore').strip()

        if msg.startswith(bs.CMD_TCP_CONNS):
            self.connections[sender_id].sendReply(
                str(self.numConnections()))
        else:
            self.process(msg, sender_id)
