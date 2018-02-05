""" Console interface for the QTGL implementation."""
try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import QWidget, QTextEdit
except ImportError:
    from PyQt4.QtCore import Qt, pyqtSignal
    from PyQt4.QtGui import QWidget, QTextEdit

from bluesky.tools.misc import cmdsplit
from . import guiio as io
from . import autocomplete


class Console(QWidget):
    lineEdit        = None
    stackText       = None

    cmdline_stacked = pyqtSignal(str, list)

    def __init__(self, parent=None):
        super(Console, self).__init__(parent)
        self.command_history = []
        self.cmd             = ''
        self.args            = []
        self.history_pos     = 0
        self.command_mem     = ''
        self.command_line    = ''
        self.initialized     = False

        # Connect to the io client's activenode changed signal
        io.actnodedata_changed.connect(self.actnodedataChanged)

    def actnodedataChanged(self, nodeid, nodedata, changed_elems):
        if 'ECHOTEXT' in changed_elems:
            self.stackText.setPlainText(nodedata.echo_text)
            self.stackText.verticalScrollBar().setValue(self.stackText.verticalScrollBar().maximum())


    def stack(self, text):
        # Add command to the command history
        self.command_history.append(text)
        self.echo(text)
        # Send stack command to sim process
        io.send_event(b'STACKCMD', text)
        self.cmdline_stacked.emit(self.cmd, self.args)
        # reset commandline and the autocomplete history
        self.setCmdline('')
        autocomplete.reset()

    def echo(self, text):
        actdata = io.get_nodedata()
        actdata.echo(text)
        self.stackText.append(text)
        self.stackText.verticalScrollBar().setValue(self.stackText.verticalScrollBar().maximum())

    def appendCmdline(self, text):
        self.setCmdline(self.command_line + text)

    def setCmdline(self, text):
        if self.command_line == text:
            return

        # if not self.initialized:
        #     self.initialized = True
        #     self.lineEdit.setHtml('>>')

        actdata = io.get_nodedata()

        self.command_line   = text
        self.cmd, self.args = cmdsplit(self.command_line)

        hintline = ''
        allhints = actdata.stack_help
        if allhints:
            hint = allhints.get(self.cmd.upper())
            if hint:
                if len(self.args) > 0:
                    hintargs = hint.split(',')
                    hintline = ' ' + str.join(',', hintargs[len(self.args):])
                else:
                    hintline = ' ' + hint

        self.lineEdit.setHtml('>>' + self.command_line + '<font color="#aaaaaa">' + hintline + '</font>')

    def keyPressEvent(self, event):
        # Enter-key: enter command
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            if len(self.command_line) > 0:
                # emit a signal with the command for the simulation thread
                self.stack(self.command_line)
                # Clear any shape command preview on the radar display
                # self.radarwidget.previewpoly(None)
                return

        newcmd = self.command_line
        if event.key() == Qt.Key_Backspace:
            newcmd = newcmd[:-1]

        elif event.key() == Qt.Key_Up:
            if self.history_pos == 0:
                self.command_mem = newcmd
            if len(self.command_history) >= self.history_pos + 1:
                self.history_pos += 1
                newcmd = self.command_history[-self.history_pos]

        elif event.key() == Qt.Key_Down:
            if self.history_pos > 0:
                self.history_pos -= 1
                if self.history_pos == 0:
                    newcmd = self.command_mem
                else:
                    newcmd = self.command_history[-self.history_pos]

        elif event.key() == Qt.Key_Tab:
            if len(newcmd) > 0:
                newcmd, displaytext = autocomplete.complete(newcmd)
                if len(displaytext) > 0:
                    self.echo(displaytext)

        elif event.key() >= Qt.Key_Space and event.key() <= Qt.Key_AsciiTilde:
            newcmd += str(event.text())#.upper()

        else:
            super(Console, self).keyPressEvent(event)
            return

        # Final processing of the command line
        self.setCmdline(newcmd)


class Cmdline(QTextEdit):
    def __init__(self, parent=None):
        super(Cmdline, self).__init__(parent)
        Console.lineEdit = self
        self.setFocusPolicy(Qt.NoFocus)


class Stackwin(QTextEdit):
    def __init__(self, parent=None):
        super(Stackwin, self).__init__(parent)
        Console.stackText = self
        self.setFocusPolicy(Qt.NoFocus)
