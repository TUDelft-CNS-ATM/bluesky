try:
    from PyQt4.QtCore import Qt, QEvent, QTimer
    from PyQt4.QtGui import QColor, QApplication, QFileDialog
    from PyQt4.QtOpenGL import QGLFormat
    print('Using Qt4 for windows and widgets')
except ImportError:
    from PyQt5.QtCore import Qt, QEvent, QTimer
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QApplication, QFileDialog
    from PyQt5.QtOpenGL import QGLFormat
    print('Using Qt5 for windows and widgets')
import numpy as np

# Local imports
from ..radarclick import radarclick
from mainwindow import MainWindow, Splash
from uievents import PanZoomEvent, ACDataEvent, StackTextEvent, PanZoomEventType, ACDataEventType, SimInfoEventType,  \
                     StackTextEventType, ShowDialogEventType, DisplayFlagEventType, RouteDataEventType, DisplayShapeEventType
from radarwidget import RadarWidget
from nd import ND
import autocomplete as ac
from ...tools.misc import cmdsplit

usage_hints = { 'CRE' : 'acid,type,lat,lon,hdg,alt,spd',
                'POS' : 'acid',
                'MOVE': 'acid,lat,lon,[alt],[hdg],[spd],[vspd]',
                'DEL': 'acid',
                'ALT': 'acid,alt',
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
                'ADDWPT': 'acid,wpname/latlon,[alt],[spd],[afterwp]',
                'DELWPT': 'acid,wpname',
                'DIRECT': 'acid,wpname',
                'LISTRTE': 'acid,[pagenr]',
                'ND': 'acid',
                'NAVDISP': 'acid',
                'NOISE': 'ON/OFF',
                'LINE': 'color,lat1,lon1,lat2,lon2',
                'ENG': 'acid',
                'DATAFEED': 'CONNECT,address,port'
                }


class Gui(QApplication):
    modes = ['Init', 'Operate', 'Hold', 'End']

    def __init__(self, navdb):
        super(Gui, self).__init__([])
        self.acdata          = ACDataEvent()
        self.navdb           = navdb
        self.radarwidget     = []
        self.command_history = []
        self.cmdargs         = []
        self.history_pos     = 0
        self.command_mem     = ''
        self.command_line    = ''
        self.prev_cmdline    = ''
        self.simevent_target = 0
        self.mousepos        = (0, 0)
        self.prevmousepos    = (0, 0)

        # Register our custom pan/zoom event
        for etype in [PanZoomEventType, ACDataEventType, SimInfoEventType,
                      StackTextEventType, ShowDialogEventType,
                      DisplayFlagEventType, RouteDataEventType,
                      DisplayShapeEventType]:
            reg_etype = QEvent.registerEventType(etype)
            if reg_etype != etype:
                print('Warning: Registered event type differs from requested type id (%d != %d)' % (reg_etype, etype))

        self.splash = Splash()
        self.splash.show()

        self.splash.showMessage('Constructing main window')
        self.processEvents()

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

        # Create the main window
        self.radarwidget = RadarWidget(navdb)
        self.win = MainWindow(self, self.radarwidget)
        self.nd  = ND(shareWidget=self.radarwidget)

        # Enable HiDPI support
        self.setAttribute(Qt.AA_UseHighDpiPixmaps)
        # Share GL context between GL widgets (i.e., widgets share buffers, textures, ...)
        self.setAttribute(Qt.AA_ShareOpenGLContexts)

        timer = QTimer(self)
        timer.timeout.connect(self.radarwidget.updateGL)
        timer.timeout.connect(self.nd.updateGL)
        timer.start(50)

    def setSimEventTarget(self, obj):
        self.simevent_target = obj

    def start(self):
        self.win.show()
        self.splash.showMessage('Done!')
        self.processEvents()
        self.splash.finish(self.win)
        self.exec_()

    def notify(self, receiver, event):
        # Keep track of event processing
        event_processed = False

        # Events from the simulation thread
        if receiver is self:
            if event.type() == PanZoomEventType:
                if event.panzoom_type() == PanZoomEvent.Zoom:
                    event.vorigin = (self.radarwidget.width / 2, self.radarwidget.height / 2)

                if event.panzoom_type() == PanZoomEvent.Pan:
                    event.value = (2.0 * event.value[0] / (self.radarwidget.zoom * self.radarwidget.ar),
                                   2.0 * event.value[1] / (self.radarwidget.zoom * self.radarwidget.flat_earth))

                # send the pan/zoom event to the radarwidget
                receiver = self.radarwidget

            elif event.type() == ACDataEventType:
                self.acdata = event
                self.radarwidget.update_aircraft_data(event)
                return True

            elif event.type() == RouteDataEventType:
                self.radarwidget.update_route_data(event)
                return True

            elif event.type() == DisplayShapeEventType:
                self.radarwidget.updatePolygon(event.name, event.data)

            elif event.type() == SimInfoEventType:
                self.win.siminfoLabel.setText('<b>F</b> = %.2f Hz, <b>sim_dt</b> = %.2f, <b>sim_t</b> = %.1f, <b>n_aircraft</b> = %d, <b>mode</b> = %s'
                    % (event.sys_freq, event.simdt, event.simt, event.n_ac, self.modes[event.mode]))
                return True

            elif event.type() == StackTextEventType:
                self.display_stack(event.text)
                return True

            elif event.type() == ShowDialogEventType:
                if event.dialog_type == event.filedialog_type:
                    self.show_file_dialog()
                return True

            elif event.type() == DisplayFlagEventType:
                # Switch/toggle/cycle radar screen features e.g. from SWRAD command
                # Coastlines
                if event.switch == "GEO":
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
                    self.nd.setWindowTitle(event.argument)
                    self.nd.setVisible(not self.nd.isVisible())

                return True

        # Mouse/trackpad event handling for the Radar widget
        if receiver is self.radarwidget:
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

                    return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Zoom, zoom, origin))
                # For touchpad scroll (2D) is used for panning
                else:
                    try:
                        dlat =  0.01 * event.pixelDelta().y() / (self.radarwidget.zoom * self.radarwidget.ar)
                        dlon = -0.01 * event.pixelDelta().x() / (self.radarwidget.zoom * self.radarwidget.flat_earth)
                        return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Pan, (dlat, dlon)))
                    except:
                        pass
            # For touchpad, pinch gesture is used for zoom
            elif event.type() == QEvent.Gesture:
                origin = (0, 0)
                zoom   = 1.0
                for g in event.gestures():
                    if g.gestureType() == Qt.PinchGesture:
                        origin = (g.centerPoint().x(), g.centerPoint().y())
                        zoom  *= g.scaleFactor() / g.lastScaleFactor()

                return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Zoom, zoom, origin))

            elif event.type() == QEvent.MouseButtonPress:
                event_processed = True
                # For mice we pan with control/command and mouse movement. Mouse button press marks the beginning of a pan
                if event.button() & Qt.RightButton:
                    self.prevmousepos = (event.x(), event.y())

                else:
                    lat, lon  = self.radarwidget.pixelCoordsToLatLon(event.x(), event.y())
                    tostack, todisplay = radarclick(self.command_line, lat, lon, self.acdata, self.navdb)
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
                event_processed = True
                self.mousepos = (event.x(), event.y())
                if event.buttons() & Qt.RightButton:
                    pan = (0.003 * (event.y() - self.prevmousepos[1]), 0.003 * (self.prevmousepos[0] - event.x()))
                    self.prevmousepos = (event.x(), event.y())
                    return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Pan, pan))

        # Other events
        if event.type() == QEvent.KeyPress:
            event_processed = True
            if event.modifiers() & Qt.ShiftModifier:
                dlat = 1.0  / (self.radarwidget.zoom * self.radarwidget.ar)
                dlon = 1.0  / (self.radarwidget.zoom * self.radarwidget.flat_earth)
                if event.key() == Qt.Key_Up:
                    return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Pan, (dlat, 0.0)))
                elif event.key() == Qt.Key_Down:
                    return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Pan, (-dlat, 0.0)))
                elif event.key() == Qt.Key_Left:
                    return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Pan, (0.0, -dlon)))
                elif event.key() == Qt.Key_Right:
                    return super(Gui, self).notify(self.radarwidget, PanZoomEvent(PanZoomEvent.Pan, (0.0, dlon)))

            elif event.key() == Qt.Key_Backspace:
                self.command_line = self.command_line[:-1]

            if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
                if len(self.command_line) > 0:
                    # emit a signal with the command for the simulation thread
                    self.command_history.append(self.command_line)
                    self.stack(self.command_line)
                    self.command_line = ''
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
                    newcmd, displaytext = ac.complete(self.command_line)
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
            self.cmdargs = cmdsplit(self.command_line)

            hint = ''
            if len(self.cmdargs) > 0:
                if self.cmdargs[0] in usage_hints:
                    hint = usage_hints[self.cmdargs[0]]
                    if len(self.cmdargs) > 1:
                        hintargs = hint.split(',')
                        hint = ' ' + str.join(',', hintargs[len(self.cmdargs)-1:])

            self.win.lineEdit.setHtml('<font color="#00ff00">>>' + self.command_line + '</font><font color="#aaaaaa">' + hint + '</font>')
            self.prev_cmdline = self.command_line

        if self.mousepos != self.prevmousepos and len(self.cmdargs) >= 3:
            self.prevmousepos = self.mousepos
            try:
                if self.cmdargs[0] == 'AREA':
                    data = np.zeros(4, dtype=np.float32)
                    data[0:2] = self.radarwidget.pixelCoordsToLatLon(self.mousepos[0], self.mousepos[1])
                    data[2] = float(self.cmdargs[1])
                    data[3] = float(self.cmdargs[2])
                    self.radarwidget.previewpoly(self.cmdargs[0], data)
                elif self.cmdargs[0] in ['BOX', 'POLY', 'POLYGON', 'CIRCLE', 'LINE']:
                    data = np.zeros(len(self.cmdargs), dtype=np.float32)
                    data[0:2] = self.radarwidget.pixelCoordsToLatLon(self.mousepos[0], self.mousepos[1])
                    for i in range(2, len(self.cmdargs), 2):
                        data[i]     = float(self.cmdargs[i])
                        data[i + 1] = float(self.cmdargs[i+1])
                    self.radarwidget.previewpoly(self.cmdargs[0], data)

            except ValueError:
                pass

        event.accept()
        return True

    def stack(self, text):
        self.postEvent(self.simevent_target, StackTextEvent(text))

    def display_stack(self, text):
        self.win.stackText.setTextColor(QColor(0, 255, 0))
        self.win.stackText.insertHtml('<br>' + text)
        self.win.stackText.verticalScrollBar().setValue(self.win.stackText.verticalScrollBar().maximum())

    def show_file_dialog(self):
        response = QFileDialog.getOpenFileName(self.win, 'Open file', 'data/scenario', 'Scenario files (*.scn)')
        if type(response) is tuple:
            fname = response[0]
        else:
            fname = response
        if len(fname) > 0:
            self.stack('IC ' + str(fname))

    def __del__(self):
        # Make sure to Clean up at quit event
        self.aboutToQuit.connect(self.win.cleanUp)
