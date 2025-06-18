""" Console interface for the QTGL implementation."""
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QTextCursor
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QWidget, QTextEdit

import bluesky as bs
from bluesky.core.signal import Signal
from bluesky.stack.cmdparser import Command
from bluesky.tools import cachefile
from bluesky.tools.misc import cmdsplit
from bluesky.network import subscribe
from bluesky.network.sharedstate import ActData, get
from bluesky.ui.qtgl import autocomplete
from bluesky.ui.radarclick import radarclick

import html

cmdline_stacked = Signal('cmdline_stacked')


def get_cmd():
    """ Return the current command in the console's command line."""
    if not Console._instance:
        return ''
    return Console._instance.cmd


def get_cmdline():
    """ Return the current text in the console's command line."""
    if not Console._instance:
        return ''
    return Console._instance.command_line


def get_args():
    """ Return the current command arguments in the console's command line."""
    if not Console._instance:
        return []
    return Console._instance.args


def process_cmdline(cmdlines):
    assert Console._instance is not None, 'No console created yet: can only change' + \
        ' command line after main window is created.'
    lines = cmdlines.split('\n')
    if lines:
        Console._instance.append_cmdline(lines[-1])
        for cmd in lines[:-1]:
            Console._instance.stack(cmd)


class Console(QWidget):
    lineEdit: 'Cmdline' = None
    stackText: QTextEdit = None
    _instance = None

    # Per-remote data
    echotext: ActData[list] = ActData()
    echoflags: ActData[list] = ActData()
    cmddict: ActData[dict] = ActData(group='stackcmds')

    def __init__(self, parent=None):
        super().__init__(parent)
        with cachefile.openfile('console_history.p') as cache:
            try:
                self.command_history = cache.load()
            except:
                self.command_history = []
        self.cmd = ''
        self.args = []
        self.history_pos = 0
        self.command_mem = ''
        self.command_line = ''

        # Connect to the io client's activenode changed signal
        Signal('CMDLINE').connect(self.set_cmdline)
        Signal('actnode-changed').connect(self.actnodeChanged)
        Signal('radarclick').connect(self.on_radarclick)

        assert Console._instance is None, "Console constructor: console instance " + \
            "already exists! Cannot have more than one console."
        Console._instance = self

        # Connect function to save command history on quit
        QApplication.instance().aboutToQuit.connect(self.close)

        # Connect to stack command list SharedState
        subscribe('STACKCMDS')

    def close(self):
        ''' Save command history when BlueSky closes. '''
        with cachefile.openfile('console_history.p') as cache:
            cache.dump(self.command_history)

    def actnodeChanged(self, nodeid):
        text = ('<br>'.join(self.echotext)).replace('\n', '<br>') + '<br>'
        self.stackText.setHtml(text)
        self.stackText.verticalScrollBar().setValue(
            self.stackText.verticalScrollBar().maximum())

    def stack(self, text):
        # Add command to the command history
        self.command_history.append(text)
        self.echo(text)
        bs.stack.stack(text)
        cmdline_stacked.emit(self.cmd, self.args)
        # reset commandline and the autocomplete history
        self.set_cmdline('')
        autocomplete.reset()
        self.history_pos = 0

    def echo(self, text, flags=None):
        cursor = self.stackText.textCursor()
        cursor.clearSelection()
        cursor.movePosition(cursor.MoveOperation.End)
        self.stackText.setTextCursor(cursor)
        self.stackText.insertHtml(text.replace('\n', '<br>') + '<br>')
        self.stackText.verticalScrollBar().setValue(
            self.stackText.verticalScrollBar().maximum())

    def on_radarclick(self, lat, lon):
        actdata = get()
        # TODO routedata isn't really a sharedstate, it only gives a selected route
        tostack, tocmdline = radarclick(get_cmdline(), lat, lon,
                                        actdata.acdata, getattr(actdata, 'routedata', None))

        process_cmdline((tostack + '\n' + tocmdline) if tostack else tocmdline)


    def append_cmdline(self, text):
        self.set_cmdline(self.command_line + text)

    def set_cmdline(self, text, cursorpos=None):
        if self.command_line == text:
            return

        self.command_line = text
        self.cmd, self.args = cmdsplit(self.command_line)
        self.cmd = self.cmd.upper()

        hintline = ''
        guicmd = Command.cmddict.get(self.cmd)
        hint = guicmd.brief[len(self.cmd) + 1:] if guicmd else self.cmddict.get(self.cmd)
        if hint:
            if len(self.args) > 0:
                hintargs = hint.split(',')
                hintline = ' ' + str.join(',', hintargs[len(self.args):])
            else:
                hintline = ' ' + hint

        self.lineEdit.set_cmdline(self.command_line, hintline, cursorpos)

    def keyPressEvent(self, event):
        ''' Handle keyboard input for bluesky. '''
        # Enter-key: enter command
        if event.key() == Qt.Key.Key_Enter or event.key() == Qt.Key.Key_Return:
            if self.command_line:
                # emit a signal with the command for the simulation thread
                self.stack(self.command_line)
                # Clear any shape command preview on the radar display
                # self.radarwidget.previewpoly(None)
                return

        newcmd = self.command_line
        cursorpos = None
        if event.key() == Qt.Key.Key_Left:
            self.lineEdit.cursor_left(
                word_level=event.modifiers() & Qt.KeyboardModifier.ControlModifier,
                select=event.modifiers() & Qt.KeyboardModifier.ShiftModifier,
            )
        elif event.key() == Qt.Key.Key_Right:
            self.lineEdit.cursor_right(
                word_level=event.modifiers() & Qt.KeyboardModifier.ControlModifier,
                select=event.modifiers() & Qt.KeyboardModifier.ShiftModifier,
            )
        elif event.key() == Qt.Key.Key_Home:
            self.lineEdit.cursor_home(
                select=event.modifiers() & Qt.KeyboardModifier.ShiftModifier,
            )
        elif event.key() == Qt.Key.Key_End:
            self.lineEdit.cursor_end(
                select=event.modifiers() & Qt.KeyboardModifier.ShiftModifier,
            )
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_V:
                pos = self.lineEdit.cursor_pos()
                clipboardText = QApplication.clipboard().text()
                newcmd = newcmd[:pos] + clipboardText + newcmd[pos:]
                cursorpos = pos + len(clipboardText)
            elif event.key() == Qt.Key.Key_X:
                QApplication.clipboard().setText(self.command_line)
                newcmd = ""
                cursorpos = 0
            elif event.key() == Qt.Key.Key_C:
                QApplication.clipboard().setText(self.command_line)
        elif event.key() >= Qt.Key.Key_Space and event.key() <= Qt.Key.Key_AsciiTilde:
            pos = self.lineEdit.cursor_pos()
            newcmd = newcmd[:pos] + event.text() + newcmd[pos:]
            # Update the cursor position with the length of the added text
            cursorpos = pos + len(event.text())
        elif event.key() == Qt.Key.Key_Backspace:
            pos = self.lineEdit.cursor_pos()
            newcmd = newcmd[:pos - 1] + newcmd[pos:]
            cursorpos = pos - 1
        elif event.key() == Qt.Key.Key_Delete:
            cursor = self.lineEdit.textCursor()
            anchor = cursor.anchor()-2
            pos = cursor.position()-2

            begin = min(anchor, pos)
            end = max(anchor, pos)
            if begin == end:
                end += 1

            if begin < len(newcmd):
                newcmd = newcmd[:begin] + newcmd[end:]
                cursorpos = begin
        elif event.key() == Qt.Key.Key_Tab:
            if newcmd:
                newcmd, displaytext = autocomplete.complete(newcmd)
                if displaytext:
                    self.echo(displaytext)
        elif not event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier |
                                        Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier):
            if event.key() == Qt.Key.Key_Up:
                if self.history_pos == 0:
                    self.command_mem = newcmd
                if len(self.command_history) >= self.history_pos + 1:
                    self.history_pos += 1
                    newcmd = self.command_history[-self.history_pos]
            elif event.key() == Qt.Key.Key_Down:
                if self.history_pos > 0:
                    self.history_pos -= 1
                    if self.history_pos == 0:
                        newcmd = self.command_mem
                    else:
                        newcmd = self.command_history[-self.history_pos]
            else:
                # Remaining keys are things like sole modifier keys, and function keys
                super().keyPressEvent(event)
        else:
            event.ignore()
            return

        # Final processing of the command line
        self.set_cmdline(newcmd, cursorpos)


class Word:
    def __init__(self, w='', type='str'):
        self.word = w

    def append(self, w):
        self.word = self.word + w
        # TODO do checks

    def __str__(self):
        ''' Word with color markup. '''
        return ''


class Cmdline(QTextEdit):
    ''' Wrapper class for the command line. '''

    def __init__(self, parent=None):
        super().__init__(parent)
        Console.lineEdit = self
        self.cmdline = ''
        # self.setFocusPolicy(Qt.NoFocus)
        self.set_cmdline('')

    def keyPressEvent(self, event):
        event.ignore()

    def set_cmdline(self, cmdline, hints='', cursorpos=None):
        ''' Set the command line with possible hints. '''
        self.setHtml('>>' +
                    html.escape(cmdline).replace(' ', '&nbsp;') +
                    '<font color="#aaaaaa">' +
                    html.escape(hints).replace(' ', '&nbsp;') +
                    '</font>'
        )
        self.cmdline = cmdline
        cursor = self.textCursor()
        if cursorpos is None:
            cursorpos = len(cmdline)
        cursor.setPosition(cursorpos + 2)
        self.setTextCursor(cursor)
        # TODO: word objects with possible list of checker functions?

    def cursor_pos(self):
        ''' Get the cursor position. '''
        return self.textCursor().position() - 2

    def cursor_left(self, word_level: bool = False, select: bool = False):
        ''' Move the cursor one position to the left. '''
        cursor = self.textCursor()
        if word_level:
            pos = cursor.position() - 2
            while pos > 0 and self.cmdline[pos - 1] == ' ':
                pos -= 1 # skip initial spaces
            while pos > 0 and self.cmdline[pos - 1] != ' ':
                pos -= 1
            pos += 2
        else:
            pos = max(2, cursor.position() - 1)

        self.cursor_set_pos(pos, select)

    def cursor_right(self, word_level: bool = False, select: bool = False):
        ''' Move the cursor one position to the right. '''
        cursor = self.textCursor()
        if word_level:
            pos = cursor.position() - 2
            while pos < len(self.cmdline) and self.cmdline[pos] == ' ':
                pos += 1 # skip initial spaces
            while pos < len(self.cmdline) and self.cmdline[pos] != ' ':
                pos += 1
            pos += 2
        else:
            pos = min(len(self.cmdline) + 2, cursor.position() + 1)

        self.cursor_set_pos(pos, select)

    def cursor_home(self, select: bool = False):
        self.cursor_set_pos(2, select)

    def cursor_end(self, select: bool = False):
        self.cursor_set_pos(len(self.cmdline) + 2, select)

    def cursor_set_pos(self, pos, select: bool=False):
        cursor = self.textCursor()
        if select:
            cursor.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)
        else:
            cursor.setPosition(pos)
        self.setTextCursor(cursor)


class Stackwin(QTextEdit):
    ''' Wrapper class for the stack output textbox. '''

    def __init__(self, parent=None):
        super().__init__(parent)
        Console.stackText = self
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def mousePressEvent(self, e):
        self.anchor = self.anchorAt(e.pos())
        if self.anchor:
            QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, e):
        if self.anchor:
            QDesktopServices.openUrl(QUrl(self.anchor))
            QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
            self.anchor = None
