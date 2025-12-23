'''
    Example external client for BlueSky.

    When you run this file, make sure python knows where to find BlueSky:

    PYTHONPATH=/path/to/your/bluesky python textclient.py
'''
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLabel, QTextEdit

import bluesky as bs
from bluesky.core import Base
from bluesky.network import subscriber
from bluesky.network.client import Client
from bluesky.ui.qtgl.console import Cmdline


# The echo textbox, command line, and bluesky network client as globals
echobox = None
cmdline = None


class Echobox(QTextEdit, Base):
    ''' Text box to show echoed text coming from BlueSky. '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setReadOnly(True)

    @subscriber
    def echo(self, text, flags=None):
        ''' Add text to this echo box. '''
        self.append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class InfoLine(QLabel, Base):
    @subscriber
    def acdata(self, data):
        ''' Example subscriber to aircraft state data '''
        self.setText(f"There are {len(data.lat)} aircraft in the simulation.")


if __name__ == '__main__':
    # Construct the Qt main object
    app = QApplication([])

    # Start the bluesky network client
    bs.init(mode='client')
    client = Client()
    network_timer = QTimer()
    network_timer.timeout.connect(client.update)
    network_timer.start(20)
    client.connect()

    # Create a window with a stack text box and a command line
    win = QWidget()
    win.setWindowTitle('Example external client for BlueSky')
    layout = QVBoxLayout()
    win.setLayout(layout)

    echobox = Echobox(win)
    cmdline = Cmdline(win)
    infoline = InfoLine(win)
    layout.addWidget(echobox)
    layout.addWidget(cmdline)
    layout.addWidget(infoline)
    win.show()

    cmdline.commandEntered.connect(bs.stack.stack)
    cmdline.commandEntered.connect(echobox.echo)
    # Let echobox act as screen object
    # NOTE: this approach will soon be deprecated
    bs.scr = echobox

    # Start the Qt main loop
    app.exec()
