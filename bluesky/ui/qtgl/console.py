try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import QWidget, QTextEdit
except ImportError:
    from PyQt4.QtCore import Qt, pyqtSignal
    from PyQt4.QtGui import QWidget, QTextEdit

import autocomplete
from ...tools.misc import cmdsplit
from ...sim.qtgl import MainManager as manager
from ...sim.qtgl import StackTextEvent


usage_hints = { 'BATCH': 'filename',
                'CRE' : 'acid,type,lat,lon,hdg,alt,spd',
                'POS' : 'acid',
                'SSD' : 'acid/ALL/OFF',
                'MOVE': 'acid,lat,lon,[alt],[hdg],[spd],[vspd]',
                'DELAY': 'time offset [secs], CMD + CMD-specific arguments',
                'DEL': 'acid',
                'ALT': 'acid,alt,[vspd]',
                'HDG': 'acid,hdg',
                'SPD': 'acid,spd',
                'NOM': 'acid',
                'VS': 'acid,vs',
                'ORIG': 'acid,apt',
                'DEST': 'acid,apt',
                'ZOOM': 'in/out',
                'PAN': 'LEFT/RIGHT/UP/DOWN/acid/airport/navid',
                'IC': 'IC/filename',
                'SAVEIC': 'filename',
                'SCHEDULE': 'execution time [secs], CMD + CMD-specific arguments',
                'DT': 'dt',
                'AREA': 'lat0,lon0,lat1,lon1,[lowalt]',
                'BOX': 'boxname,lat0,lon0,lat1,lon1',
                'POLY': 'polyname,lat0,lon0,lat1,lon1,lat2,lon2,...',
                'TAXI': 'ON/OFF',
                'SWRAD': 'GEO/GRID/APT/VOR/WPT/LABEL/TRAIL,[dt]/[value]',
                'TRAIL': 'ON/OFF,[delta_t]',
                'MCRE': 'n,type/*,alt/*,spd/*,dest/*',
                'DIST': 'lat1,lon1,lat2,lon2',
                'LNAV': 'acid,ON/OFF',
                'VNAV': 'acid,ON/OFF',
                'ASAS': 'acid,ON/OFF',
                'ADDWPT': 'acid,wpname/latlon/fly-by/fly-over,[alt],[spd],[afterwp]',
                'DELWPT': 'acid,wpname',
                'DIRECT': 'acid,wpname',
                'LISTRTE': 'acid,[pagenr]',
                'ND': 'acid',
                'NAVDISP': 'acid',
                'NOISE': 'ON/OFF',
                'LINE': 'name,lat1,lon1,lat2,lon2',
                'ENG': 'acid',
                'DATAFEED': 'ON/OFF'
                }


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

    def stack(self, text):
        # Add command to the command history
        self.command_history.append(text)
        # Send stack command to sim process
        manager.sendEvent(StackTextEvent(cmdtext=text))
        self.cmdline_stacked.emit(self.cmd, self.args)
        # reset commandline and the autocomplete history
        self.setCmdline('')
        autocomplete.reset()

    def echo(self, text):
        self.stackText.append(text)
        self.stackText.verticalScrollBar().setValue(self.stackText.verticalScrollBar().maximum())

    def appendCmdline(self, text):
        self.setCmdline(self.command_line + text)

    def setCmdline(self, text):
        if self.command_line == text:
            return

        self.command_line   = text
        self.cmd, self.args = cmdsplit(self.command_line)

        hint = ''
        if self.cmd in usage_hints:
            hint = usage_hints[self.cmd]
            if len(self.args) > 0:
                hintargs = hint.split(',')
                hint = ' ' + str.join(',', hintargs[len(self.args):])

        self.lineEdit.setHtml('>>' + self.command_line + '<font color="#aaaaaa">' + hint + '</font>')

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
                    self.display_stack(displaytext)

        elif event.key() >= Qt.Key_Space and event.key() <= Qt.Key_AsciiTilde:
            newcmd += str(event.text()).upper()

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
