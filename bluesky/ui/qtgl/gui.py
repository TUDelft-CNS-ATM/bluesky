""" QTGL Gui for BlueSky."""
try:
    from PyQt5.QtCore import Qt, QEvent, QTimer, QT_VERSION, QT_VERSION_STR
    from PyQt5.QtWidgets import QApplication, QFileDialog

except ImportError:
    from PyQt4.QtCore import Qt, QEvent, QTimer, QT_VERSION, QT_VERSION_STR
    from PyQt4.QtGui import QApplication, QFileDialog

# Local imports
from bluesky.tools.misc import tim2txt
from bluesky import settings

from bluesky.ui.qtgl.customevents import ACDataEvent, RouteDataEvent
from bluesky.ui.qtgl.mainwindow import MainWindow
from bluesky.ui.qtgl.docwindow import DocWindow
from bluesky.ui.qtgl.radarwidget import RadarWidget
from bluesky.ui.qtgl.infowindow import InfoWindow
from bluesky.ui.qtgl.nd import ND

from . import guiio as io

print(('Using Qt ' + QT_VERSION_STR + ' for windows and widgets'))

# Register settings defaults
settings.set_variable_defaults(scenario_path='scenario')

class Gui(QApplication):
    modes = ['Init', 'Operate', 'Hold', 'End']

    def __init__(self):
        super(Gui, self).__init__([])
        self.acdata          = ACDataEvent()
        self.routedata       = RouteDataEvent()
        self.radarwidget     = None
        self.win             = None
        self.nd              = None
        self.docwin          = None
        self.initialized     = False

        # Enable HiDPI support (Qt5 only)
        if QT_VERSION >= 0x050000:
            self.setAttribute(Qt.AA_UseHighDpiPixmaps)

        # Connect to BlueSky sim io
        io.event_received.connect(self.on_simevent_received)
        io.stream_received.connect(self.on_simstream_received)

    def init(self):
        ''' Create the main window and related widgets. '''
        self.radarwidget = RadarWidget()
        self.win         = MainWindow(self, self.radarwidget)
        self.nd          = ND(shareWidget=self.radarwidget)
        # self.infowin     = InfoWindow()
        try:
            self.docwin      = DocWindow(self)
        except Exception as e:
            print('Couldnt make docwindow:', e)
        # self.aman = AMANDisplay()
        gltimer          = QTimer(self)
        gltimer.timeout.connect(self.radarwidget.updateGL)
        gltimer.timeout.connect(self.nd.updateGL)
        gltimer.start(50)

        self.win.show()
        # self.infowin.show()
        # self.infowin.addPlotTab()
        # for i in range(10):
            # self.infowin.plottab.addPlot()
        self.initialized = True

    def start(self):
        self.exec_()

    def quit(self):
        # Send quit to server
        io.send_event(b'QUIT')
        self.closeAllWindows()

    def on_simevent_received(self, eventname, eventdata, sender_id):
        ''' Processing of events from simulation nodes. '''
        # initialization order problem: TODO
        if not self.initialized:
            return

        elif eventname == b'CMDLINE':
            self.win.console.set_cmdline(eventdata)

        # ND window for selected aircraft
        elif eventname == b'SHOWND':
            if eventdata:
                self.nd.setAircraftID(eventdata)
            self.nd.setVisible(not self.nd.isVisible())

        elif eventname == b'SHOWDIALOG':
            dialog = eventdata.get('dialog')
            args   = eventdata.get('args')
            if dialog == 'OPENFILE':
                self.show_file_dialog()
            elif dialog == 'DOC':
                self.show_doc_window(args)

    def on_simstream_received(self, streamname, data, sender_id):
        if not self.initialized:
            return

        if streamname == b'SIMINFO':
            speed, simdt, simt, simtclock, ntraf, state, scenname = data
            simt = tim2txt(simt)[:-3]
            simtclock = tim2txt(simtclock)[:-3]
            self.win.setNodeInfo(sender_id, simt, scenname)
            if sender_id == io.actnode():
                self.win.siminfoLabel.setText(u'<b>t:</b> %s, <b>\u0394t:</b> %.2f, <b>Speed:</b> %.1fx, <b>UTC:</b> %s, <b>Mode:</b> %s, <b>Aircraft:</b> %d, <b>Conflicts:</b> %d/%d, <b>LoS:</b> %d/%d'
                    % (simt, simdt, speed, simtclock, self.modes[state], ntraf, self.acdata.nconf_cur, self.acdata.nconf_tot, self.acdata.nlos_cur, self.acdata.nlos_tot))

        elif streamname == b'ACDATA':
            self.acdata = ACDataEvent(data)
            self.radarwidget.update_aircraft_data(self.acdata)
            if self.nd.ac_id in self.acdata.id:
                idx = self.acdata.id.index(self.nd.ac_id.upper())
                lat = self.acdata.lat[idx]
                lon = self.acdata.lon[idx]
                trk = self.acdata.trk[idx]
                tas = self.acdata.tas[idx]
                self.nd.update_aircraft_data(idx, lat, lon, tas, trk, len(self.acdata.lat))

        elif streamname == b'ROUTEDATA':
            self.routedata = RouteDataEvent(data)
            self.radarwidget.update_route_data(self.routedata)

    def notify(self, receiver, event):
        # Send all key presses directly to the main window
        if event.type() == QEvent.KeyPress:
            return self.win.keyPressEvent(event)

        return super(Gui, self).notify(receiver, event)

    def show_file_dialog(self):
        response = QFileDialog.getOpenFileName(self.win, 'Open file', settings.scenario_path, 'Scenario files (*.scn)')
        if type(response) is tuple:
            fname = response[0]
        else:
            fname = response
        if len(fname) > 0:
            self.win.console.stack('IC ' + str(fname))

    def show_doc_window(self, cmd=''):
        self.docwin.show_cmd_doc(cmd)
        self.docwin.show()

    def __del__(self):
        # Make sure to Clean up at quit event
        self.aboutToQuit.connect(self.win.cleanUp)
