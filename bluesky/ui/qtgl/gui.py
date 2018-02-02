""" QTGL Gui for BlueSky."""
try:
    from PyQt5.QtCore import Qt, QEvent, QTimer, QT_VERSION, QT_VERSION_STR
    from PyQt5.QtWidgets import QApplication, QFileDialog

except ImportError:
    from PyQt4.QtCore import Qt, QEvent, QTimer, QT_VERSION, QT_VERSION_STR
    from PyQt4.QtGui import QApplication, QFileDialog

# Local imports
from bluesky.ui.radarclick import radarclick
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

# Qt smaller than 5.6.2 needs a different approach to pinch gestures
correct_pinch = False
if QT_VERSION <= 0x050600:
    import platform
    correct_pinch = platform.system() == 'Darwin'

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
        self.mousedragged    = False
        self.mousepos        = (0, 0)
        self.prevmousepos    = (0, 0)
        self.panzoomchanged  = False
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
            self.win.console.setCmdline(eventdata)

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

        # TODO stack init
        # elif event.type() == StackInitEventType:
        #     self.win.console.addStackHelp(sender_id, event.stackdict)


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
        # Keep track of event processing
        event_processed = False

        # Mouse/trackpad event handling for the Radar widget
        if receiver is self.radarwidget and self.radarwidget.initialized:
            panzoom = None
            if event.type() == QEvent.Wheel:
                # For mice we zoom with control/command and the scrolwheel
                if event.modifiers() & Qt.ControlModifier:
                    origin = (event.pos().x(), event.pos().y())
                    zoom   = 1.0
                    try:
                        if event.pixelDelta():
                            # High resolution scroll
                            zoom *= (1.0 + 0.01 * event.pixelDelta().y())
                        else:
                            # Low resolution scroll
                            zoom *= (1.0 + 0.001 * event.angleDelta().y())
                    except:
                        zoom *= (1.0 + 0.001 * event.delta())
                    self.panzoomchanged = True
                    return self.radarwidget.panzoom(zoom=zoom, origin=origin)

                # For touchpad scroll (2D) is used for panning
                else:
                    try:
                        dlat =  0.01 * event.pixelDelta().y() / (self.radarwidget.zoom * self.radarwidget.ar)
                        dlon = -0.01 * event.pixelDelta().x() / (self.radarwidget.zoom * self.radarwidget.flat_earth)
                        self.panzoomchanged = True
                        return self.radarwidget.panzoom(pan=(dlat, dlon))
                    except:
                        pass

            # For touchpad, pinch gesture is used for zoom
            elif event.type() == QEvent.Gesture:
                zoom   = None
                pan    = None
                dlat   = 0.0
                dlon   = 0.0
                for g in event.gestures():
                    if g.gestureType() == Qt.PinchGesture:
                        if zoom is None:
                            zoom = 1.0
                        zoom  *= g.scaleFactor()
                        if correct_pinch:
                            zoom /= g.lastScaleFactor()
                    elif g.gestureType() == Qt.PanGesture:
                        if abs(g.delta().y() + g.delta().x()) > 1e-1:
                            dlat += 0.005 * g.delta().y() / (self.radarwidget.zoom * self.radarwidget.ar)
                            dlon -= 0.005 * g.delta().x() / (self.radarwidget.zoom * self.radarwidget.flat_earth)
                            pan = (dlat, dlon)
                if pan is not None or zoom is not None:
                    self.panzoomchanged = True
                    return self.radarwidget.panzoom(pan, zoom, self.mousepos)

            elif event.type() == QEvent.MouseButtonPress and event.button() & Qt.LeftButton:
                event_processed   = True
                self.mousedragged = False
                # For mice we pan with control/command and mouse movement. Mouse button press marks the beginning of a pan
                self.prevmousepos = (event.x(), event.y())

            elif event.type() == QEvent.MouseButtonRelease and event.button() & Qt.LeftButton and not self.mousedragged:
                event_processed = True
                lat, lon  = self.radarwidget.pixelCoordsToLatLon(event.x(), event.y())
                tostack, tocmdline = radarclick(self.win.console.command_line, lat, lon, self.acdata, self.routedata)
                if len(tocmdline) > 0:
                    if '\n' in tocmdline:
                        self.win.console.setCmdline('')
                        # Clear any shape command preview on the radar display
                        self.radarwidget.previewpoly(None)
                    else:
                        self.win.console.appendCmdline(tocmdline)
                    if len(tostack) > 0:
                        self.win.console.stack(tostack)

            elif event.type() == QEvent.MouseMove:
                event_processed   = True
                self.mousedragged = True
                self.mousepos = (event.x(), event.y())
                if event.buttons() & Qt.LeftButton:
                    dlat = 0.003 * (event.y() - self.prevmousepos[1]) / (self.radarwidget.zoom * self.radarwidget.ar)
                    dlon = 0.003 * (self.prevmousepos[0] - event.x()) / (self.radarwidget.zoom * self.radarwidget.flat_earth)
                    self.prevmousepos = (event.x(), event.y())
                    self.panzoomchanged = True
                    return self.radarwidget.panzoom(pan=(dlat, dlon))

            # Update pan/zoom to simulation thread only when the pan/zoom gesture is finished
            elif (event.type() == QEvent.MouseButtonRelease or event.type() == QEvent.TouchEnd) and self.panzoomchanged:
                self.panzoomchanged = False
                io.send_event(b'PANZOOM', dict(pan=(self.radarwidget.panlat, self.radarwidget.panlon),
                                               zoom=self.radarwidget.zoom, ar=self.radarwidget.ar, absolute=True))

        # Send all key presses directly to the main window
        if event.type() == QEvent.KeyPress:
            self.win.keyPressEvent(event)
            return True

        # If we haven't processed the event: call Base Class Method to Continue Normal Event Processing
        if not event_processed:
            return super(Gui, self).notify(receiver, event)
        cmd = self.win.console.cmd.upper()
        if cmd in ['AREA', 'BOX', 'POLY', 'POLYALT', 'POLYGON', 'CIRCLE', 'LINE']:
            if self.mousepos != self.prevmousepos and len(self.win.console.args) >= 2:
                self.prevmousepos = self.mousepos
                try:
                    # get the largest even number of points
                    start = 0 if cmd == 'AREA' else 3 if cmd == 'POLYALT' else 1
                    end   = ((len(self.win.console.args) - start) // 2) * 2 + start
                    data  = [float(v) for v in self.win.console.args[start:end]]
                    data += self.radarwidget.pixelCoordsToLatLon(*self.mousepos)
                    self.radarwidget.previewpoly(cmd, data)

                except ValueError:
                    pass

        event.accept()
        return True

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
