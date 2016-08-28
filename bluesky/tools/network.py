from ..settings import gui
from .. import stack
import time
import socket
import threading

if gui == 'qtgl':
    try:
        from PyQt5.QtCore import pyqtSlot
        from PyQt5.QtNetwork import QTcpServer, QTcpSocket
    except ImportError:
        from PyQt4.QtCore import pyqtSlot
        from PyQt4.QtNetwork import QTcpServer, QTcpSocket


    class TcpSocket(QTcpSocket):
        def __init__(self, parent=None):
            super(TcpSocket, self).__init__(parent)
            if parent is None:
                self.error.connect(self.onError)
                self.connected.connect(self.onConnected)
                self.disconnected.connect(self.onDisconnected)

        @pyqtSlot()
        def onError(self):
            print self.socket.errorString()

        @pyqtSlot()
        def onConnected(self):
            print 'TcpClient connected'

        @pyqtSlot()
        def onDisconnected(self):
            print 'TcpClient disconnected'

        def isConnected(self):
            return (self.state() == self.ConnectedState)

        @pyqtSlot()
        def onReadyRead(self):
            self.processData(self.readAll())

        def processData(self, data):
            # Placeholder function; override it with your own implementation
            print 'TcpSocket received', data


    class TcpServer(QTcpServer):
        def __init__(self, parent=None):
            super(TcpServer, self).__init__(parent)
            self.connections = list()

        def incomingConnection(self, socketDescriptor):
            newconn = TcpSocket(self)
            newconn.setSocketDescriptor(socketDescriptor)
            newconn.readyRead.connect(self.onReadyRead)
            self.connections.append(newconn)

        @pyqtSlot()
        def onReadyRead(self):
            sender_id = self.connections.index(self.sender())
            data      = self.sender().readAll()
            self.processData(sender_id, data)

        def processData(self, sender_id, data):
            # Placeholder function; override it with your own implementation
            print 'TcpServer received', data, 'from sender no', sender_id


elif gui == 'pygame':



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
                print "Server connected. HOST: %s, PORT: %s" % (ip, port)
            except Exception, err:
                self.is_connected = False
                print "Connection Error: %s" % err
                pass

        def disconnectFromHost(self):
            try:
                self.sock.close()
                self.is_connected = False
                print "Server disconnected."
            except Exception, err:
                print "Disconnection Error: %s" % err
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
                    self.parse_data(data)
                    time.sleep(0.1)
                except Exception, err:
                    print "Revecier Error: %s" % err
                    time.sleep(1.0)

        def processData(self, data):
            # rewrite this function
            print "parsing data..."

    class TcpServer(object):
        def __init__(self):
            pass

        def start(self):
            pass

        def processData(self, sender_id, data):
            pass


class StackTelnetServer(TcpServer):
    def __init__(self):
        super(StackTelnetServer, self).__init__()

    def processData(self, sender_id, data):
        stack.stack(str(data).strip())
