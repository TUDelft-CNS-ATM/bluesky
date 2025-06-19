'''
    Example external client for BlueSky.

    When you run this file, make sure python knows where to find BlueSky:

    PYTHONPATH=/path/to/your/bluesky python textclient.py
'''
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLabel, QTextEdit
from PyQt6.QtGui import QTextCursor, QKeyEvent, QTextCharFormat, QColor

import bluesky as bs
from bluesky.core import Base
from bluesky.network import subscriber
from bluesky.network.client import Client
from bluesky.stack import stack


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


class Cmdline(QTextEdit):
    ''' Wrapper class for the command line. '''

    PROMPT = ">> "

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(21)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.hint_text = ""
        self.insertPrompt()
        self.textChanged.connect(self._updateHint)

    def insertPrompt(self):
        self._updateText(self.PROMPT, "", "")
        self.moveCursor(QTextCursor.MoveOperation.End)

    def keyPressEvent(self, event: QKeyEvent):
        cursor = self.textCursor()
        current_text = self._getInputText()

        if (
            event.key() == Qt.Key.Key_C
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and not self.textCursor().hasSelection()
        ):
            # Clear command on ctrl-c only if nothing is selected.
            # Else, handle as copy shortcut by default QTextEdit implementation.
            self.insertPrompt()
            return

        if event.key() == Qt.Key.Key_Home:
            move_mode = (
                QTextCursor.MoveMode.KeepAnchor
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                else QTextCursor.MoveMode.MoveAnchor
            )
            cursor.setPosition(len(self.PROMPT), move_mode)
            self.setTextCursor(cursor)
            return

        if event.key() == Qt.Key.Key_End:
            move_mode = (
                QTextCursor.MoveMode.KeepAnchor
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                else QTextCursor.MoveMode.MoveAnchor
            )
            cursor.setPosition(len(self.PROMPT + current_text), move_mode)
            self.setTextCursor(cursor)
            return

        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            command = current_text.strip()
            if command:
                stack(command)
                echobox.echo(command)
            self.insertPrompt()
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

    def _ensureCursorAtValidPos(self):
        min_pos = len(self.PROMPT)
        max_pos = len(self.PROMPT + self._getInputText())

        cursor = self.textCursor()
        anchor = cursor.anchor()
        position = cursor.position()

        anchor = min(max(anchor, min_pos), max_pos)
        position = min(max(position, min_pos), max_pos)

        cursor.setPosition(anchor, QTextCursor.MoveMode.MoveAnchor)
        cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)

        self.setTextCursor(cursor)

    def _getInputText(self):
        text = self.toPlainText()
        return (
            text[len(self.PROMPT) : len(text) - len(self.hint_text)]
            if text.startswith(self.PROMPT) and text.endswith(self.hint_text)
            else ""
        )

    def _updateHint(self):
        current_input = self._getInputText()
        self.hint_text = self._getHint(current_input.strip())
        cursor_position = self.textCursor().position()

        self._updateText(self.PROMPT, current_input, self.hint_text)

        cursor = self.textCursor()
        min_pos = len(self.PROMPT)
        max_pos = len(self.PROMPT + current_input)
        cursor.setPosition(min(max(cursor_position, min_pos), max_pos))
        self.setTextCursor(cursor)

    def _updateText(self, prompt, command, hint):
        # blocking signals to prevent recursion
        self.blockSignals(True)
        self.clear()
        cursor = self.textCursor()

        fmt_blue = QTextCharFormat()
        fmt_blue.setForeground(QColor("blue"))

        fmt_hint = QTextCharFormat()
        fmt_hint.setForeground(QColor("lightgray"))

        cursor.setCharFormat(fmt_blue)
        cursor.insertText(prompt + command)

        cursor.setCharFormat(fmt_hint)
        cursor.insertText(hint)
        self.blockSignals(False)

    def _getHint(self, input_text):
        if input_text == "":
            return ""
        elif input_text == "help":
            return " (shows help menu)"
        elif input_text.startswith("load"):
            return " <filename>"
        return ""


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

    # Let echobox act as screen object
    # NOTE: this approach will soon be deprecated
    bs.scr = echobox

    # Start the Qt main loop
    app.exec()
