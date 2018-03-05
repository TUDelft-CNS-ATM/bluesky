""" Main window for the QTGL gui."""
import platform
import os
try:
    from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QItemSelectionModel, QSize
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import QMainWindow, QSplashScreen, QTreeWidgetItem, \
        QPushButton, QFileDialog
    from PyQt5 import uic
except ImportError:
    from PyQt4.QtCore import Qt, pyqtSlot, QTimer, QSize
    from PyQt4.QtGui import QPixmap, QMainWindow, QIcon, QSplashScreen, \
        QItemSelectionModel, QTreeWidgetItem, QPushButton, QFileDialog
    from PyQt4 import uic

# Local imports
from bluesky import settings
from bluesky.tools.misc import tim2txt
from bluesky.ui.qtgl import guiio as io

# Child windows
from bluesky.ui.qtgl.docwindow import DocWindow
from bluesky.ui.qtgl.radarwidget import RadarWidget
from bluesky.ui.qtgl.infowindow import InfoWindow
from bluesky.ui.qtgl.nd import ND

is_osx = platform.system() == 'Darwin'

# Register settings defaults
settings.set_variable_defaults(gfx_path='data/graphics',
                               stack_text_color=(0, 255, 0),
                               stack_background_color=(102, 102, 102))

fg = settings.stack_text_color
bg = settings.stack_background_color

class Splash(QSplashScreen):
    """ Splash screen: BlueSky logo during start-up"""
    def __init__(self):
        super(Splash, self).__init__(QPixmap(os.path.join(settings.gfx_path, 'splash.gif')), Qt.WindowStaysOnTopHint)


class MainWindow(QMainWindow):
    """ Qt window process: from .ui file read UI window-definition of main window """

    modes = ['Init', 'Hold', 'Operate', 'End']

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.radarwidget = RadarWidget()
        self.nd = ND(shareWidget=self.radarwidget)
        # self.infowin = InfoWindow()
        # self.infowin.show()
        # self.infowin.addPlotTab()
        # for i in range(10):
        # self.infowin.plottab.addPlot()

        try:
            self.docwin = DocWindow(self)
        except Exception as e:
            print('Couldnt make docwindow:', e)
        # self.aman = AMANDisplay()
        gltimer = QTimer(self)
        gltimer.timeout.connect(self.radarwidget.updateGL)
        gltimer.timeout.connect(self.nd.updateGL)
        gltimer.start(50)
        self.app = app
        if is_osx:
            self.app.setWindowIcon(QIcon(os.path.join(settings.gfx_path, 'bluesky.icns')))
        else:
            self.app.setWindowIcon(QIcon(os.path.join(settings.gfx_path, 'icon.gif')))

        uic.loadUi(os.path.join(settings.gfx_path, 'mainwindow.ui'), self)

        # list of buttons to connect to, give icons, and tooltips
        #           the button         the icon      the tooltip    the callback
        buttons = { self.zoomin :     ['zoomin.svg', 'Zoom in', self.buttonClicked],
                    self.zoomout :    ['zoomout.svg', 'Zoom out', self.buttonClicked],
                    self.panleft :    ['panleft.svg', 'Pan left', self.buttonClicked],
                    self.panright :   ['panright.svg', 'Pan right', self.buttonClicked],
                    self.panup :      ['panup.svg', 'Pan up', self.buttonClicked],
                    self.pandown :    ['pandown.svg', 'Pan down', self.buttonClicked],
                    self.ic :         ['stop.svg', 'Initial condition', self.buttonClicked],
                    self.op :         ['play.svg', 'Operate', self.buttonClicked],
                    self.hold :       ['pause.svg', 'Hold', self.buttonClicked],
                    self.fast :       ['fwd.svg', 'Enable fast-time', self.buttonClicked],
                    self.fast10 :     ['ffwd.svg', 'Fast-forward 10 seconds', self.buttonClicked],
                    self.sameic :     ['frwd.svg', 'Restart same IC', self.buttonClicked],
                    self.showac :     ['AC.svg', 'Show/hide aircraft', self.buttonClicked],
                    self.showpz :     ['PZ.svg', 'Show/hide PZ', self.buttonClicked],
                    self.showapt :    ['apt.svg', 'Show/hide airports', self.buttonClicked],
                    self.showwpt :    ['wpt.svg', 'Show/hide waypoints', self.buttonClicked],
                    self.showlabels : ['lbl.svg', 'Show/hide text labels', self.buttonClicked],
                    self.showmap :    ['geo.svg', 'Show/hide satellite image', self.buttonClicked],
                    self.shownodes :  ['nodes.svg', 'Show/hide node list', self.buttonClicked]}

        for b in buttons.items():
            # Set icon
            if not b[1][0] is None:
                icon = QIcon(os.path.join(settings.gfx_path, 'icons/' + b[1][0]))
                b[0].setIcon(icon)
            # Set tooltip
            if not b[1][1] is None:
                b[0].setToolTip(b[1][1])
            # Connect clicked signal
            b[0].clicked.connect(b[1][2])

        # Link menubar buttons
        self.action_Open.triggered.connect(self.show_file_dialog)
        self.action_Save.triggered.connect(self.buttonClicked)
        self.actionBlueSky_help.triggered.connect(self.show_doc_window)

        self.radarwidget.setParent(self.centralwidget)
        self.verticalLayout.insertWidget(0, self.radarwidget, 1)
        # Connect to io client's nodelist changed signal
        io.nodes_changed.connect(self.nodesChanged)
        io.actnodedata_changed.connect(self.actnodedataChanged)
        io.event_received.connect(self.on_simevent_received)
        io.stream_received.connect(self.on_simstream_received)

        self.nodetree.setVisible(False)
        self.nodetree.setIndentation(0)
        self.nodetree.setColumnCount(2)
        self.nodetree.setStyleSheet('padding:0px')
        self.nodetree.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.nodetree.header().resizeSection(0, 130)
        self.nodetree.itemClicked.connect(self.nodetreeClicked)
        self.maxhostnum = 0
        self.hosts      = dict()
        self.nodes      = dict()

        fgcolor = '#%02x%02x%02x' % fg
        bgcolor = '#%02x%02x%02x' % bg

        self.stackText.setStyleSheet('color:' + fgcolor + '; background-color:' + bgcolor)
        self.lineEdit.setStyleSheet('color:' + fgcolor + '; background-color:' + bgcolor)

        self.nconf_cur = self.nconf_tot = self.nlos_cur = self.nlos_tot = 0

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier \
                and event.key() in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            dlat = 1.0 / (self.radarwidget.zoom * self.radarwidget.ar)
            dlon = 1.0 / (self.radarwidget.zoom * self.radarwidget.flat_earth)
            if event.key() == Qt.Key_Up:
                self.radarwidget.panzoom(pan=(dlat, 0.0))
            elif event.key() == Qt.Key_Down:
                self.radarwidget.panzoom(pan=(-dlat, 0.0))
            elif event.key() == Qt.Key_Left:
                self.radarwidget.panzoom(pan=(0.0, -dlon))
            elif event.key() == Qt.Key_Right:
                self.radarwidget.panzoom(pan=(0.0, dlon))

        elif event.key() == Qt.Key_Escape:
            self.app.quit()

        elif event.key() == Qt.Key_F11:  # F11 = Toggle Full Screen mode
            if not self.isFullScreen():
                self.showFullScreen()
            else:
                self.showNormal()

        else:
            # All other events go to the BlueSky console
            return self.console.keyPressEvent(event)

        return True

    def closeEvent(self, event):
        self.app.quit()
        return True

    def actnodedataChanged(self, nodeid, nodedata, changed_elems):
        node = self.nodes[nodeid]
        self.nodelabel.setText('<b>Node</b> {}:{}'.format(node.host_num, node.node_num))
        self.nodetree.setCurrentItem(node, 0, QItemSelectionModel.ClearAndSelect)

    def nodesChanged(self, data):
        for host_id, host_data in data.items():
            host = self.hosts.get(host_id)
            if not host:
                host = QTreeWidgetItem(self.nodetree)
                self.maxhostnum += 1
                host.host_num = self.maxhostnum
                host.host_id = host_id
                hostname = 'This computer' if host_id == io.get_hostid() else str(host_id)
                f = host.font(0)
                f.setBold(True)
                host.setExpanded(True)
                btn = QPushButton(self.nodetree)
                btn.host_id = host_id
                btn.setText(hostname)
                btn.setFlat(True)
                btn.setStyleSheet('font-weight:bold')

                btn.setIcon(QIcon(os.path.join(settings.gfx_path, 'icons/addnode.svg')))
                btn.setIconSize(QSize(24, 16))
                btn.setLayoutDirection(Qt.RightToLeft)
                btn.setMaximumHeight(16)
                btn.clicked.connect(self.buttonClicked)
                self.nodetree.setItemWidget(host, 0, btn)
                self.hosts[host_id] = host

            for node_num, node_id in enumerate(host_data['nodes']):
                if node_id not in self.nodes:
                    # node_num = node_id[-2] * 256 + node_id[-1]
                    node = QTreeWidgetItem(host)
                    node.setText(0, '{}:{} <init>'.format(host.host_num, node_num + 1))
                    node.setText(1, '00:00:00')
                    node.node_id  = node_id
                    node.node_num = node_num + 1
                    node.host_num = host.host_num

                    self.nodes[node_id] = node

    def on_simevent_received(self, eventname, eventdata, sender_id):
        ''' Processing of events from simulation nodes. '''
        # ND window for selected aircraft
        if eventname == b'SHOWND':
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
        if streamname == b'SIMINFO':
            speed, simdt, simt, simtclock, ntraf, state, scenname = data
            simt = tim2txt(simt)[:-3]
            simtclock = tim2txt(simtclock)[:-3]
            self.setNodeInfo(sender_id, simt, scenname)
            if sender_id == io.actnode():
                self.siminfoLabel.setText(u'<b>t:</b> %s, <b>\u0394t:</b> %.2f, <b>Speed:</b> %.1fx, <b>UTC:</b> %s, <b>Mode:</b> %s, <b>Aircraft:</b> %d, <b>Conflicts:</b> %d/%d, <b>LoS:</b> %d/%d'
                    % (simt, simdt, speed, simtclock, self.modes[state], ntraf, self.nconf_cur, self.nconf_tot, self.nlos_cur, self.nlos_tot))
        elif streamname == b'ACDATA':
            self.nconf_cur = data['nconf_cur']
            self.nconf_tot = data['nconf_tot']
            self.nlos_cur = data['nlos_cur']
            self.nlos_tot = data['nlos_tot']

    def setNodeInfo(self, connid, time, scenname):
        node = self.nodes.get(connid)
        if node:
            node.setText(0, '{}:{} <{}>'.format(node.host_num, node.node_num, scenname))
            node.setText(1, time)

    @pyqtSlot(QTreeWidgetItem, int)
    def nodetreeClicked(self, item, column):
        if item in self.hosts.values():
            item.setSelected(False)
            item.child(0).setSelected(True)
            io.actnode(item.child(0).node_id)
        else:
            io.actnode(item.node_id)


    @pyqtSlot()
    def buttonClicked(self):
        actdata = io.get_nodedata()
        if self.sender() == self.shownodes:
            vis = not self.nodetree.isVisible()
            self.nodetree.setVisible(vis)
            self.shownodes.setText('>' if vis else '<')
        if self.sender() == self.zoomin:
            self.radarwidget.panzoom(zoom=1.4142135623730951)
        elif self.sender() == self.zoomout:
            self.radarwidget.panzoom(zoom=0.70710678118654746)
        elif self.sender() == self.pandown:
            self.radarwidget.panzoom(pan=(-0.5,  0.0))
        elif self.sender() == self.panup:
            self.radarwidget.panzoom(pan=( 0.5,  0.0))
        elif self.sender() == self.panleft:
            self.radarwidget.panzoom(pan=( 0.0, -0.5))
        elif self.sender() == self.panright:
            self.radarwidget.panzoom(pan=( 0.0,  0.5))
        elif self.sender() == self.ic:
            self.app.show_file_dialog()
        elif self.sender() == self.sameic:
            io.send_event(b'STACKCMD', 'IC IC')
        elif self.sender() == self.hold:
            io.send_event(b'STACKCMD', 'HOLD')
        elif self.sender() == self.op:
            io.send_event(b'STACKCMD', 'OP')
        elif self.sender() == self.fast:
            io.send_event(b'STACKCMD', 'FF')
        elif self.sender() == self.fast10:
            io.send_event(b'STACKCMD', 'FF 0:0:10')
        elif self.sender() == self.showac:
            actdata.show_traf = not actdata.show_traf
        elif self.sender() == self.showpz:
            actdata.show_pz = not actdata.show_pz
        elif self.sender() == self.showapt:
            if actdata.show_apt < 3:
                actdata.show_apt += 1
            else:
                actdata.show_apt = 0
        elif self.sender() == self.showwpt:
            if actdata.show_wpt < 2:
                actdata.show_wpt += 1
            else:
                actdata.show_wpt = 0
        elif self.sender() == self.showlabels:
            actdata.show_lbl -= 1
            if actdata.show_lbl < 0:
                actdata.show_lbl = 2
        elif self.sender() == self.showmap:
            actdata.show_map = not actdata.show_map
        elif self.sender() == self.action_Save:
            io.send_event(b'STACKCMD', 'SAVEIC')
        elif hasattr(self.sender(), 'host_id'):
            print(self.sender())
            io.send_event(b'ADDNODES', 1)

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
