""" Console interface for the QTGL implementation."""
from PyQt6.QtCore import QChildEvent, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QTextCursor, QKeyEvent, QTextCharFormat, QColor
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

cmdline_stacked = Signal('cmdline_stacked')


def get_cmdline():
    """ Return the current text in the console's command line without prompt and hint."""
    if not Console._instance:
        return ''
    return Console._instance.command_line


def get_cmd():
    """ Return the current command in the console's command line."""
    cmd, _ = cmdsplit(get_cmdline())
    return cmd


def get_args():
    """ Return the current command arguments in the console's command line."""
    _, args = cmdsplit(get_cmdline())
    return args


def process_cmdline(cmdlines):
    assert Console._instance is not None, 'No console created yet: can only change' + \
        ' command line after main window is created.'
    lines = cmdlines.split('\n')
    if lines:
        Console._instance.append_cmdline(lines[-1])
        for cmd in lines[:-1]:
            Console._instance.lineEdit.appendHistory(cmd)
            Console._instance.stack(cmd)


class Console(QWidget):
    lineEdit: 'Cmdline' = None
    stackText: QTextEdit = None
    _instance = None

    # Per-remote data
    echotext: ActData[list] = ActData()
    echoflags: ActData[list] = ActData()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Connect to the io client's activenode changed signal
        Signal('CMDLINE').connect(self.set_cmdline)
        Signal('actnode-changed').connect(self.actnodeChanged)
        Signal('radarclick').connect(self.on_radarclick)

        assert Console._instance is None, "Console constructor: console instance " + \
            "already exists! Cannot have more than one console."
        Console._instance = self

        # Connect to stack command list SharedState
        subscribe('STACKCMDS')

    @property
    def command_line(self):
        return self.lineEdit.getInputText()

    def childEvent(self, event: QChildEvent | None) -> None:
        if (
            event
            and isinstance(event.child(), Cmdline)
            and event.type()
            in (QChildEvent.Type.ChildAdded, QChildEvent.Type.ChildPolished)
        ):
            self.lineEdit = event.child()
            self.lineEdit.commandEntered.connect(self.stack)
            self.lineEdit.tabCompletion.connect(self.tabCompletion)
        return super().childEvent(event)

    def actnodeChanged(self, nodeid):
        text = ('<br>'.join(self.echotext)).replace('\n', '<br>') + '<br>'
        self.stackText.setHtml(text)
        self.stackText.verticalScrollBar().setValue(
            self.stackText.verticalScrollBar().maximum()
        )

    def stack(self, text):
        # Output command to stack pane, add to command stack, and inform subscribers
        self.echo(text)
        bs.stack.stack(text)
        cmd, args = cmdsplit(self.command_line)
        cmdline_stacked.emit(cmd, args)

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
            self.stackText.verticalScrollBar().maximum()
        )

    def on_radarclick(self, lat, lon):
        actdata = get()
        # TODO routedata isn't really a sharedstate, it only gives a selected route
        tostack, tocmdline = radarclick(self.command_line, lat, lon,
                                        actdata.acdata, getattr(actdata, 'routedata', None))

        process_cmdline((tostack + '\n' + tocmdline) if tostack else tocmdline)

    def append_cmdline(self, text):
        self.lineEdit.append(text)

    def set_cmdline(self, text):
        self.lineEdit.setText(text, reposition_cursor=True)

    def tabCompletion(self, newcmd):
        if newcmd:
            newcmd, displaytext = autocomplete.complete(newcmd)
            self.set_cmdline(newcmd)
            if displaytext:
                self.echo(displaytext)


class Cmdline(QTextEdit):
    """Wrapper class for the command line."""

    PROMPT = ">> "

    cmddict: ActData[dict] = ActData(group="stackcmds")

    commandEntered = pyqtSignal(str)
    tabCompletion = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(21)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        with cachefile.openfile("console_history.p") as cache:
            try:
                self.command_history = cache.load()
            except:
                self.command_history = []
        self.history_pos = 0
        self.command_mem = ""
        # Connect function to save command history on quit
        QApplication.instance().aboutToQuit.connect(self._cache_history)

        self.hint_text = ""
        self.clear()
        self.textChanged.connect(self.setText)

    def _cache_history(self):
        """Save command history when BlueSky closes."""
        with cachefile.openfile("console_history.p") as cache:
            cache.dump(self.command_history)

    def clear(self):
        self.history_pos = 0
        self._updateText(self.PROMPT, "", "")
        self.moveCursor(QTextCursor.MoveOperation.End)

    def keyPressEvent(self, event: QKeyEvent):
        current_text = self.getInputText().strip()

        if (
            event.key() == Qt.Key.Key_C
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and not self.textCursor().hasSelection()
        ):
            # Clear command on ctrl-c only if nothing is selected.
            # Else, handle as copy shortcut by default QTextEdit implementation.
            self.clear()
        elif event.key() == Qt.Key.Key_Home:
            self._moveCursorBegin(event.modifiers())
        elif event.key() == Qt.Key.Key_End:
            self._moveCursorEnd(event.modifiers())
        elif event.key() == Qt.Key.Key_Up:
            self._historyPrevious()
        elif event.key() == Qt.Key.Key_Down:
            self._historyNext()
        elif event.key() == Qt.Key.Key_Tab:
            self.tabCompletion.emit(current_text)
        elif event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            command = current_text
            if command:
                # Add command to the command history
                self.appendHistory(command)
                self.commandEntered.emit(command)
            self.clear()
        else:
            super().keyPressEvent(event)

        self._ensureCursorAtValidPos()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._ensureCursorAtValidPos()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._ensureCursorAtValidPos()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._ensureCursorAtValidPos()

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self._ensureCursorAtValidPos()

    def _moveCursorBegin(self, modifiers):
        cursor = self.textCursor()
        move_mode = (
            QTextCursor.MoveMode.KeepAnchor
            if modifiers & Qt.KeyboardModifier.ShiftModifier
            else QTextCursor.MoveMode.MoveAnchor
        )
        cursor.setPosition(len(self.PROMPT), move_mode)
        self.setTextCursor(cursor)

    def _moveCursorEnd(
        self, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
    ):
        cursor = self.textCursor()
        move_mode = (
            QTextCursor.MoveMode.KeepAnchor
            if modifiers & Qt.KeyboardModifier.ShiftModifier
            else QTextCursor.MoveMode.MoveAnchor
        )
        cursor.setPosition(len(self.PROMPT + self.getInputText()), move_mode)
        self.setTextCursor(cursor)

    def _historyPrevious(self):
        if self.history_pos == 0:
            self.command_mem = self.getInputText()
        if len(self.command_history) >= self.history_pos + 1:
            self.history_pos += 1
            newcmd = self.command_history[-self.history_pos]
            self.setText(newcmd, reposition_cursor=True)

    def _historyNext(self):
        if self.history_pos > 0:
            self.history_pos -= 1
            if self.history_pos == 0:
                newcmd = self.command_mem
            else:
                newcmd = self.command_history[-self.history_pos]
            self.setText(newcmd, reposition_cursor=True)

    def _ensureCursorAtValidPos(self):
        min_pos = len(self.PROMPT)
        max_pos = len(self.PROMPT + self.getInputText())

        cursor = self.textCursor()
        anchor = cursor.anchor()
        position = cursor.position()

        anchor = min(max(anchor, min_pos), max_pos)
        position = min(max(position, min_pos), max_pos)

        cursor.setPosition(anchor, QTextCursor.MoveMode.MoveAnchor)
        cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)

        self.setTextCursor(cursor)

    def appendHistory(self, text: str) -> None:
        if len(text.strip()) == 0:
            return
        self.command_history.append(text)
        self.history_pos = 0

    def getInputText(self):
        text = self.toPlainText()
        return (
            text[len(self.PROMPT) : len(text) - len(self.hint_text)]
            if text.startswith(self.PROMPT) and text.endswith(self.hint_text)
            else ""
        )

    def append(self, text: str | None) -> None:
        if text is None:
            return
        self.setText(self.getInputText() + text, reposition_cursor=True)
        

    def setText(self, text: str | None = None, *, reposition_cursor=False) -> None:
        if text is None:
            text = self.getInputText()
        self.hint_text = self._getHint(text.strip())
        cursor_position = self.textCursor().position()

        self._updateText(self.PROMPT, text, self.hint_text)

        if reposition_cursor:
            self._moveCursorEnd()
        else:
            cursor = self.textCursor()
            min_pos = len(self.PROMPT)
            max_pos = len(self.PROMPT + text)
            cursor.setPosition(min(max(cursor_position, min_pos), max_pos))
            self.setTextCursor(cursor)

    def _updateText(self, prompt, command, hint):
        self.hint_text = hint
        # blocking signals to prevent recursion
        self.blockSignals(True)
        super().clear()
        cursor = self.textCursor()

        cursor.insertText(prompt + command)

        fmt_hint = QTextCharFormat()
        fmt_hint.setForeground(QColor("#aaaaaa"))  # TODO - account for light/dark mode
        cursor.setCharFormat(fmt_hint)
        cursor.insertText(hint)
        self._ensureCursorAtValidPos()
        self.blockSignals(False)

    def _getHint(self, text):
        cmd, args = cmdsplit(text)
        cmd = cmd.upper()

        hintline = ""
        guicmd = Command.cmddict.get(cmd)
        hint = guicmd.brief[len(cmd) + 1 :] if guicmd else self.cmddict.get(cmd)
        if hint:
            if len(args) > 0:
                hintargs = hint.split(",")
                hintline = " " + str.join(",", hintargs[len(args) :])
            else:
                hintline = " " + hint
        return hintline


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
