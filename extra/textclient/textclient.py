'''
    Example external client for BlueSky.

    When you run this file, make sure python knows where to find BlueSky:

    PYTHONPATH=/path/to/your/bluesky python textclient.py
'''
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
except ImportError:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit

from bluesky.network import Client


# The echo textbox, command line, and bluesky network client as globals
echobox = None
cmdline = None
bsclient = None


class TextClient(Client):
    '''
        Subclassed Client with a timer to periodically check for incoming data,
        an overridden event function to handle data, and a stack function to
        send stack commands to BlueSky.
    '''
    def __init__(self):
        super().__init__()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)

    def event(self, name, data, sender_id):
        ''' Overridden event function to handle incoming ECHO commands. '''
        if name == b'ECHO' and echobox is not None:
            echobox.echo(**data)

    def stack(self, text):
        ''' Stack function to send stack commands to BlueSky. '''
        self.send_event(b'STACK', text)

    def echo(self, text, flags=None):
        ''' Overload Client's echo function. '''
        if echobox is not None:
            echobox.echo(text, flags)


class Echobox(QTextEdit):
    ''' Text box to show echoed text coming from BlueSky. '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.NoFocus)

    def echo(self, text, flags=None):
        ''' Add text to this echo box. '''
        self.append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class Cmdline(QTextEdit):
    ''' Wrapper class for the command line. '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(21)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def keyPressEvent(self, event):
        ''' Handle Enter keypress to send a command to BlueSky. '''
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            if bsclient is not None:
                bsclient.stack(self.toPlainText())
                echobox.echo(self.toPlainText())
            self.setText('')
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    # Construct the Qt main object
    app = QApplication([])

    # Create a window with a stack text box and a command line
    win = QWidget()
    win.setWindowTitle('Example external client for BlueSky')
    layout = QVBoxLayout()
    win.setLayout(layout)

    echobox = Echobox(win)
    cmdline = Cmdline(win)
    layout.addWidget(echobox)
    layout.addWidget(cmdline)
    win.show()

    # Create and start BlueSky client
    bsclient = TextClient()
    bsclient.connect(event_port=11000, stream_port=11001)

    # Start the Qt main loop
    app.exec()
