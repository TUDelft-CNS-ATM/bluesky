""" QTGL Gui for BlueSky."""
try:
    from PyQt5.QtCore import Qt, QEvent, QTimer, QT_VERSION, QT_VERSION_STR
    from PyQt5.QtWidgets import QApplication, QFileDialog, QErrorMessage
    from PyQt5.QtOpenGL import QGLFormat

except ImportError:
    from PyQt4.QtCore import Qt, QEvent, QTimer, QT_VERSION, QT_VERSION_STR
    from PyQt4.QtGui import QApplication, QFileDialog, QErrorMessage
    from PyQt4.QtOpenGL import QGLFormat

# Local imports
from bluesky.ui.radarclick import radarclick
from bluesky.tools.misc import tim2txt
from bluesky import settings
from bluesky.simulation.qtgl import MainManager as manager
from bluesky.simulation.qtgl import PanZoomEvent, ACDataEvent, RouteDataEvent, \
                     PanZoomEventType, ACDataEventType, SimInfoEventType,  \
                     StackTextEventType, ShowDialogEventType, \
                     DisplayFlagEventType, RouteDataEventType, \
                     DisplayShapeEventType, StackInitEventType, \
                     AMANEventType, NUMEVENTS
from .mainwindow import MainWindow, Splash
from .docwindow import DocWindow
from .radarwidget import RadarWidget
from .infowindow import InfoWindow
from .nd import ND

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
        self.mousedragged    = False
        self.mousepos        = (0, 0)
        self.prevmousepos    = (0, 0)
        self.panzoomchanged  = False
        self.simt            = 0.0

        # Register our custom pan/zoom event
        for etype in range(1000, 1000 + NUMEVENTS):
            reg_etype = QEvent.registerEventType(etype)
            if reg_etype != etype:
                print(('Warning: Registered event type differs from requested type id (%d != %d)' % (reg_etype, etype)))

        self.splash = Splash()
        self.splash.show()

        # Install error message handler
        handler = QErrorMessage.qtHandler()
        handler.setWindowFlags(Qt.WindowStaysOnTopHint)

        # Check and set OpenGL capabilities
        if not QGLFormat.hasOpenGL():
            raise RuntimeError('No OpenGL support detected for this system!')
        else:
            f = QGLFormat()
            f.setVersion(3, 3)
            f.setProfile(QGLFormat.CoreProfile)
            f.setDoubleBuffer(True)
            QGLFormat.setDefaultFormat(f)
            print(('QGLWidget initialized for OpenGL version %d.%d' % (f.majorVersion(), f.minorVersion())))

        # Enable HiDPI support (Qt5 only)
        if QT_VERSION >= 0x050000:
            self.setAttribute(Qt.AA_UseHighDpiPixmaps)

    def init(self):
        self.splash.showMessage('Constructing main window')
        self.processEvents()
        # Create the main window and related widgets
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

    def prestart(self):
        self.win.show()
        # self.infowin.show()
        # self.infowin.addPlotTab()
        # for i in range(10):
            # self.infowin.plottab.addPlot()
        self.splash.showMessage('Done!')
        self.processEvents()
        self.splash.finish(self.win)

    def start(self):
        self.prestart()
        self.exec_()

    def quit(self):
        self.closeAllWindows()

    def notify(self, receiver, event):
        # Keep track of event processing
        event_processed = False

        # Events from the simulation threads
        if receiver is self:
            if event.type() == PanZoomEventType:
                if event.zoom is not None:
                    event.origin = (self.radarwidget.width / 2, self.radarwidget.height / 2)

                if event.pan is not None and not event.absolute:
                    event.pan = (2.0 * event.pan[0] / (self.radarwidget.zoom * self.radarwidget.ar),
                                 2.0 * event.pan[1] / (self.radarwidget.zoom * self.radarwidget.flat_earth))

                # send the pan/zoom event to the radarwidget
                self.radarwidget.event(event)

            elif event.type() == ACDataEventType:
                self.acdata = event
                self.radarwidget.update_aircraft_data(event)
                if self.nd.ac_id in event.id:
                    idx = event.id.index(self.nd.ac_id.upper())
                    lat = event.lat[idx]
                    lon = event.lon[idx]
                    trk = event.trk[idx]
                    tas = event.tas[idx]
                    self.nd.update_aircraft_data(idx, lat, lon, tas, trk, len(event.lat))
                return True

            elif event.type() == RouteDataEventType:
                self.routedata = event
                self.radarwidget.update_route_data(event)
                return True

            elif event.type() == DisplayShapeEventType:
                self.radarwidget.updatePolygon(event.name, event.data)

            elif event.type() == SimInfoEventType:
                simt = tim2txt(event.simt)[:-3]
                simtclock = tim2txt(event.simtclock)[:-3]
                self.win.setNodeInfo(manager.sender()[0], simt, event.scenname)
                if manager.sender()[0] == manager.actnode():
                    self.simt = event.simt
                    self.win.siminfoLabel.setText(u'<b>t:</b> %s, <b>\u0394t:</b> %.2f, <b>Speed:</b> %.1fx, <b>UTC:</b> %s, <b>Mode:</b> %s, <b>Aircraft:</b> %d, <b>Conflicts:</b> %d/%d, <b>LoS:</b> %d/%d'
                        % (simt, event.simdt, event.sys_freq, simtclock, self.modes[event.mode], event.n_ac, self.acdata.nconf_cur, self.acdata.nconf_tot, self.acdata.nlos_cur, self.acdata.nlos_tot))
                return True

            elif event.type() == StackTextEventType:
                event_processed = True
                if event.disptext:
                    self.win.console.echo(event.disptext)
                if event.cmdtext:
                    self.win.console.setCmdline(event.cmdtext)

            elif event.type() == StackInitEventType:
                event_processed = True
                self.win.console.addStackHelp(manager.sender()[0], event.stackdict)

            elif event.type() == ShowDialogEventType:
                if event.dialog_type == event.filedialog_type:
                    self.show_file_dialog()
                elif event.dialog_type == event.docwin_type:
                    self.show_doc_window(event.cmd)
                return True

            elif event.type() == DisplayFlagEventType:
                # Switch/toggle/cycle radar screen features e.g. from SWRAD command
                if event.switch == 'RESET':
                    self.radarwidget.clearNodeData()
                # Coastlines
                elif event.switch == "GEO":
                    self.radarwidget.show_coast = not self.radarwidget.show_coast

                # FIR boundaries
                elif event.switch == "FIR":
                    self.radarwidget.showfir = not self.radarwidget.showfir

                # Airport: 0 = None, 1 = Large, 2= All
                elif event.switch == "APT":
                    self.radarwidget.show_apt = not self.radarwidget.show_apt

                # Waypoint: 0 = None, 1 = VOR, 2 = also WPT, 3 = Also terminal area wpts
                elif event.switch == "VOR" or event.switch == "WPT" or event.switch == "WP" or event.switch == "NAV":
                    self.radarwidget.show_apt = not self.radarwidget.show_apt

                # Satellite image background on/off
                elif event.switch == "SAT":
                    self.radarwidget.show_map = not self.radarwidget.show_map

                # Satellite image background on/off
                elif event.switch == "TRAF":
                    self.radarwidget.show_traf = not self.radarwidget.show_traf

                # ND window for selected aircraft
                elif event.switch == "ND":
                    self.nd.setAircraftID(event.argument)
                    self.nd.setVisible(not self.nd.isVisible())

                elif event.switch == "SSD":
                    self.radarwidget.show_ssd(event.argument)

                elif event.switch == "SYM":
                    # For now only toggle PZ
                    self.radarwidget.show_pz = not self.radarwidget.show_pz

                elif event.switch == "DEFWPT":
                    self.radarwidget.defwpt(event.argument)

                elif event.switch == "FILTERALT":
                    # First argument is an on/off flag
                    nact = self.radarwidget.nodedata[manager.sender()[0]]
                    if event.argument[0]:
                        nact.filteralt = event.argument[1:]
                    else:
                        nact.filteralt = False

                return True

            elif event.type() == AMANEventType:
                # self.aman.update(self.simt, event)
                pass

        # Mouse/trackpad event handling for the Radar widget
        elif receiver is self.radarwidget and self.radarwidget.initialized:
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
                    panzoom = PanZoomEvent(zoom=zoom, origin=origin)

                # For touchpad scroll (2D) is used for panning
                else:
                    try:
                        dlat =  0.01 * event.pixelDelta().y() / (self.radarwidget.zoom * self.radarwidget.ar)
                        dlon = -0.01 * event.pixelDelta().x() / (self.radarwidget.zoom * self.radarwidget.flat_earth)
                        panzoom = PanZoomEvent(pan=(dlat, dlon))
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
                    panzoom = PanZoomEvent(pan, zoom, self.mousepos)

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
                    panzoom = PanZoomEvent(pan=(dlat, dlon))

            # Update pan/zoom to simulation thread only when the pan/zoom gesture is finished
            elif (event.type() == QEvent.MouseButtonRelease or event.type() == QEvent.TouchEnd) and self.panzoomchanged:
                self.panzoomchanged = False
                manager.sendEvent(PanZoomEvent( pan=(self.radarwidget.panlat, self.radarwidget.panlon),
                                                zoom=self.radarwidget.zoom, absolute=True))

            # If we've just processed a change to pan and/or zoom, send the event to the radarwidget
            if panzoom is not None:
                self.panzoomchanged = True
                return self.radarwidget.event(panzoom)

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
