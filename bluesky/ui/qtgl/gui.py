try:
    from PyQt5.QtCore import Qt, QEvent, QTimer
    from PyQt5.QtWidgets import QApplication, QFileDialog, QErrorMessage
    from PyQt5.QtOpenGL import QGLFormat
    QT_VERSION = 5
    print('Using Qt5 for windows and widgets')
except ImportError:
    from PyQt4.QtCore import Qt, QEvent, QTimer
    from PyQt4.QtGui import QApplication, QFileDialog, QErrorMessage
    from PyQt4.QtOpenGL import QGLFormat
    QT_VERSION = 4
    print('Using Qt4 for windows and widgets')
import numpy as np

# Local imports
from ..radarclick import radarclick
from mainwindow import MainWindow, Splash
# from aman import AMANDisplay
from ...sim.qtgl import MainManager as manager
from ...sim.qtgl import PanZoomEvent, ACDataEvent, RouteDataEvent, StackTextEvent, \
                     PanZoomEventType, ACDataEventType, SimInfoEventType,  \
                     StackTextEventType, ShowDialogEventType, \
                     DisplayFlagEventType, RouteDataEventType, \
                     DisplayShapeEventType, \
                     AMANEventType, NUMEVENTS
from radarwidget import RadarWidget
from nd import ND
import autocomplete
from ...tools.misc import cmdsplit, tim2txt
from ...settings import scenario_path
import platform

is_osx = platform.system() == 'Darwin'


usage_hints = { 'BATCH': 'filename',
                'CRE' : 'acid,type,lat,lon,hdg,alt,spd',
                'POS' : 'acid',
                'SSD' : 'acid/ALL/OFF',
                'MOVE': 'acid,lat,lon,[alt],[hdg],[spd],[vspd]',
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


class Gui(QApplication):
    modes = ['Init', 'Operate', 'Hold', 'End']

    def __init__(self):
        super(Gui, self).__init__([])
        self.acdata          = ACDataEvent()
        self.routedata       = RouteDataEvent()
        self.navdb           = None
        self.radarwidget     = []
        self.command_history = []
        self.cmd             = ''
        self.args            = []
        self.history_pos     = 0
        self.command_mem     = ''
        self.command_line    = ''
        self.prev_cmdline    = ''
        self.mousedragged    = False
        self.mousepos        = (0, 0)
        self.prevmousepos    = (0, 0)
        self.panzoomchanged  = False
        self.simt            = 0.0

        # Register our custom pan/zoom event
        for etype in range(1000, 1000 + NUMEVENTS):
            reg_etype = QEvent.registerEventType(etype)
            if reg_etype != etype:
                print('Warning: Registered event type differs from requested type id (%d != %d)' % (reg_etype, etype))

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
            print('QGLWidget initialized for OpenGL version %d.%d' % (f.majorVersion(), f.minorVersion()))

        # Enable HiDPI support (Qt5 only)
        if QT_VERSION == 5:
            self.setAttribute(Qt.AA_UseHighDpiPixmaps)

    def init(self, navdb):
        self.splash.showMessage('Constructing main window')
        self.processEvents()
        # Create the main window and related widgets
        self.navdb       = navdb
        self.radarwidget = RadarWidget(navdb)
        self.win         = MainWindow(self, self.radarwidget)
        self.nd          = ND(shareWidget=self.radarwidget)
        # self.aman = AMANDisplay()

        gltimer          = QTimer(self)
        gltimer.timeout.connect(self.radarwidget.updateGL)
        gltimer.timeout.connect(self.nd.updateGL)
        gltimer.start(50)

    def start(self):
        self.win.show()
        self.splash.showMessage('Done!')
        self.processEvents()
        self.splash.finish(self.win)
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
                    self.display_stack(event.disptext)
                if event.cmdtext:
                    self.command_line = event.cmdtext

            elif event.type() == ShowDialogEventType:
                if event.dialog_type == event.filedialog_type:
                    self.show_file_dialog()
                return True

            elif event.type() == DisplayFlagEventType:
                # Switch/toggle/cycle radar screen features e.g. from SWRAD command
                if event.switch == 'RESET':
                    self.radarwidget.clearPolygons()
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
                        if is_osx:
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
                tostack, todisplay = radarclick(self.command_line, lat, lon, self.acdata, self.navdb, self.routedata)
                if len(todisplay) > 0:
                    if '\n' in todisplay:
                        self.command_line = ''
                        # Clear any shape command preview on the radar display
                        self.radarwidget.previewpoly(None)
                    else:
                        self.command_line += todisplay
                    if len(tostack) > 0:
                        self.command_history.append(tostack)
                        self.stack(tostack)

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
                self.sendEvent(manager.instance, PanZoomEvent(  pan=(self.radarwidget.panlat, self.radarwidget.panlon),
                                                                zoom=self.radarwidget.zoom, absolute=True))

            # If we've just processed a change to pan and/or zoom, send the event to the radarwidget
            if panzoom is not None:
                self.panzoomchanged = True
                return self.radarwidget.event(panzoom)

        # Other events
        if event.type() == QEvent.KeyPress:
            event_processed = True
            if event.modifiers() & Qt.ShiftModifier:
                dlat = 1.0  / (self.radarwidget.zoom * self.radarwidget.ar)
                dlon = 1.0  / (self.radarwidget.zoom * self.radarwidget.flat_earth)
                if event.key() == Qt.Key_Up:
                    return self.radarwidget.event(PanZoomEvent(pan=(dlat, 0.0)))
                elif event.key() == Qt.Key_Down:
                    return self.radarwidget.event(PanZoomEvent(pan=(-dlat, 0.0)))
                elif event.key() == Qt.Key_Left:
                    return self.radarwidget.event(PanZoomEvent(pan=(0.0, -dlon)))
                elif event.key() == Qt.Key_Right:
                    return self.radarwidget.event(PanZoomEvent(pan=(0.0, dlon)))

            elif event.key() == Qt.Key_Escape:
                    self.quit()

            elif event.key() == Qt.Key_Backspace:
                self.command_line = self.command_line[:-1]

            elif event.key() == Qt.Key_F11:  # F11 = Toggle Full Screen mode
                if not self.win.isFullScreen():
                    self.win.showFullScreen()
                else:
                    self.win.showNormal()

            if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
                if len(self.command_line) > 0:
                    # emit a signal with the command for the simulation thread
                    self.command_history.append(self.command_line)
                    self.stack(self.command_line)
                    self.command_line = ''
                    # Reset the autocomplete history
                    autocomplete.reset()
                    # Clear any shape command preview on the radar display
                    self.radarwidget.previewpoly(None)

            elif event.key() == Qt.Key_Up:
                if self.history_pos == 0:
                    self.command_mem = self.command_line
                if len(self.command_history) >= self.history_pos + 1:
                    self.history_pos += 1
                    self.command_line = self.command_history[-self.history_pos]

            elif event.key() == Qt.Key_Down:
                if self.history_pos > 0:
                    self.history_pos -= 1
                    if self.history_pos == 0:
                        self.command_line = self.command_mem
                    else:
                        self.command_line = self.command_history[-self.history_pos]

            elif event.key() == Qt.Key_Tab:
                if len(self.command_line) > 0:
                    newcmd, displaytext = autocomplete.complete(self.command_line)
                    self.command_line   = newcmd
                    if len(displaytext) > 0:
                        self.display_stack(displaytext)

            elif event.key() >= Qt.Key_Space and event.key() <= Qt.Key_AsciiTilde:
                self.command_line += str(event.text()).upper()

        # If we haven't processed the event: call Base Class Method to Continue Normal Event Processing
        if not event_processed:
            return super(Gui, self).notify(receiver, event)

        # Otherwise, final processing of the command line and accept the event.
        if self.command_line != self.prev_cmdline:
            self.cmd, self.args = cmdsplit(self.command_line)

            hint = ''
            if self.cmd in usage_hints:
                hint = usage_hints[self.cmd]
                if len(self.args) > 0:
                    hintargs = hint.split(',')
                    hint = ' ' + str.join(',', hintargs[len(self.args):])

            self.win.lineEdit.setHtml('>>' + self.command_line + '<font color="#aaaaaa">' + hint + '</font>')
            self.prev_cmdline = self.command_line

        if self.mousepos != self.prevmousepos and len(self.args) >= 2:
            self.prevmousepos = self.mousepos
            try:
                if self.cmd == 'AREA':
                    data = np.zeros(4, dtype=np.float32)
                    data[0:2] = self.radarwidget.pixelCoordsToLatLon(self.mousepos[0], self.mousepos[1])
                    data[2] = float(self.args[0])
                    data[3] = float(self.args[1])
                    self.radarwidget.previewpoly(self.cmd, data)
                elif self.cmd in ['BOX', 'POLY', 'POLYGON', 'CIRCLE', 'LINE']:
                    data = np.zeros(len(self.args) + 1, dtype=np.float32)
                    for i in range(1, len(self.args), 2):
                        data[i - 1] = float(self.args[i])
                        data[i]     = float(self.args[i + 1])
                    data[-2:]       = self.radarwidget.pixelCoordsToLatLon(self.mousepos[0], self.mousepos[1])
                    self.radarwidget.previewpoly(self.cmd, data)

            except:
                pass

        event.accept()
        return True

    def stack(self, text):
        self.sendEvent(manager.instance, StackTextEvent(cmdtext=text))
        # Echo back to command window
        self.display_stack(text)

    def display_stack(self, text):
        self.win.stackText.append(text)
        self.win.stackText.verticalScrollBar().setValue(self.win.stackText.verticalScrollBar().maximum())

    def show_file_dialog(self):
        response = QFileDialog.getOpenFileName(self.win, 'Open file', scenario_path, 'Scenario files (*.scn)')
        if type(response) is tuple:
            fname = response[0]
        else:
            fname = response
        if len(fname) > 0:
            self.stack('IC ' + str(fname))

    def __del__(self):
        # Make sure to Clean up at quit event
        self.aboutToQuit.connect(self.win.cleanUp)
