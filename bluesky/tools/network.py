from ..settings import gui
if gui == 'qtgl':
    try:
        from PyQt5.QtCore import QObject, pyqtSlot
        from PyQt5.QtNetwork import QTcpServer, QTcpSocket, QHostAddress
    except ImportError:
        from PyQt4.QtCore import QObject, pyqtSlot
        from PyQt4.QtNetwork import QTcpServer, QTcpSocket, QHostAddress

    class TcpClient(QObject):
        def __init__(self, socket=None):
            super(TcpClient, self).__init__()
            self.is_connected = False
            if socket is None:
                self.socket = QTcpSocket()
            else:
                self.socket = socket
            self.socket.error.connect(self.error)
            self.socket.connected.connect(self.connected)
            self.socket.disconnected.connect(self.disconnected)
            self.socket.readyRead.connect(self.readyRead)

        @pyqtSlot()
        def error(self):
            print self.socket.errorString()

        @pyqtSlot()
        def connected(self):
            self.is_connected = True
            print 'TcpClient connected'

        @pyqtSlot()
        def disconnected(self):
            self.is_connected = False
            print 'TcpClient disconnected'

        @pyqtSlot()
        def readyRead(self):
            data = self.socket.readAll()
            self.parse_data(data)

        def connectToHost(self, host, port):
            print 'Trying to connect to host', host, 'at port', port
            self.socket.connectToHost(host, port)

        def disconnectFromHost(self):
            self.socket.disconnectFromHost()

        def write(self, buf):
            self.socket.write(buf)

        def parse_data(self, data):
            # Placeholder function; override it with your own implementation
            print str(data).strip()

    class TcpServer(QObject):
        def __init__(self):
            super(TcpServer, self).__init__()
            self.server = QTcpServer()
            self.server.newConnection.connect(self.incomingConnection)

        @pyqtSlot()
        def incomingConnection(self):
            self.client = TcpClient(self.server.nextPendingConnection())
            self.client.parse_data = self.parse_data

        def start(self):
            if self.server.listen(QHostAddress.Any, 8888):
                print "Tcp server started"
            else:
                print "Error starting Tcp server"

        def parse_data(self, data):
            print 'Server: ', str(data).strip()

        def moveToThread(self, target_thread):
            self.server.moveToThread(target_thread)
            super(TcpServer, self).moveToThread(target_thread)


elif gui == 'pygame':
    import socket
    import threading
    import time

    class TcpClient(object):
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

        def receiver(self):
            while True:
                if not self.is_connected:
                    time.sleep(1)
                    continue

                try:
                    data = self.sock.recv(self.buffer_size)
                    self.parse_data(data)
                    time.sleep(0.1)
                except Exception, err:
                    print "Revecier Error: %s" % err
                    time.sleep(1)

        def parse_data(self, data):
            # rewrite this function
            print "parsing data..."

    class TcpServer(object):
        def __init__(self):
            pass

        def start(self):
            pass

        def parse_data(self, data):
            pass


class StackTelnetServer(TcpServer):
    def __init__(self, stack):
        super(StackTelnetServer, self).__init__()
        self.stack = stack

    def parse_data(self, data):
        self.stack.stack(str(data).strip())
