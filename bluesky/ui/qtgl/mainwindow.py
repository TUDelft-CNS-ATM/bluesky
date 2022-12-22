""" Main window for the QTGL gui."""
from pathlib import Path
import platform

try:
    from PyQt5.QtWidgets import QApplication as app
    from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QItemSelectionModel, QSize
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import QMainWindow, QSplashScreen, QTreeWidgetItem, \
        QPushButton, QFileDialog, QDialog, QTreeWidget, QVBoxLayout, \
        QDialogButtonBox, QMenu
    from PyQt5 import uic
except ImportError:
    from PyQt6.QtWidgets import QApplication as app
    from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QItemSelectionModel, QSize
    from PyQt6.QtGui import QPixmap, QIcon
    from PyQt6.QtWidgets import QMainWindow, QSplashScreen, QTreeWidgetItem, \
        QPushButton, QFileDialog, QDialog, QTreeWidget, QVBoxLayout, \
        QDialogButtonBox, QMenu
    from PyQt6 import uic

# Local imports
import bluesky as bs
from bluesky.pathfinder import ResourcePath
from bluesky.tools.misc import tim2txt
from bluesky.network import get_ownip
from bluesky.ui import palette

# Child windows
from bluesky.ui.qtgl.docwindow import DocWindow
from bluesky.ui.qtgl.radarwidget import RadarWidget
from bluesky.ui.qtgl.infowindow import InfoWindow
from bluesky.ui.qtgl.settingswindow import SettingsWindow
# from bluesky.ui.qtgl.nd import ND

if platform.system().lower() == "windows":
    from bluesky.ui.pygame.dialog import fileopen

# Register settings defaults
bs.settings.set_variable_defaults(gfx_path='graphics')

palette.set_default_colours(stack_text=(0, 255, 0),
                            stack_background=(102, 102, 102))

fg = palette.stack_text
bg = palette.stack_background

class Splash(QSplashScreen):
    """ Splash screen: BlueSky logo during start-up"""
    def __init__(self):
        splashfile = bs.resource(bs.settings.gfx_path) / 'splash.gif'
        super().__init__(QPixmap(splashfile.as_posix()), Qt.WindowType.WindowStaysOnTopHint)


class DiscoveryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setMinimumSize(200,200) # To prevent Geometry error
        self.hosts = []
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.serverview = QTreeWidget()
        self.serverview.setHeaderLabels(['Server', 'Ports'])
        self.serverview.setIndentation(0)
        self.serverview.setStyleSheet('padding:0px')
        self.serverview.header().resizeSection(0, 180)
        layout.addWidget(self.serverview)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(self.on_accept)
        btns.rejected.connect(parent.closeEvent)

        bs.net.server_discovered.connect(self.add_srv)

    def add_srv(self, address, ports):
        for host in self.hosts:
            if address == host.address and ports == host.ports:
                # We already know this server, skip
                return
        host = QTreeWidgetItem(self.serverview)
        host.address = address
        host.ports = ports
        host.hostname = 'This computer' if address == get_ownip() else address
        host.setText(0, host.hostname)

        host.setText(1, '{},{}'.format(*ports))
        self.hosts.append(host)

    def on_accept(self):
        host = self.serverview.currentItem()
        if host:
            bs.net.stop_discovery()
            hostname = host.address
            eport, sport = host.ports
            bs.net.connect(hostname=hostname, event_port=eport, stream_port=sport)
            self.close()


class MainWindow(QMainWindow):
    """ Qt window process: from .ui file read UI window-definition of main window """

    modes = ['Init', 'Hold', 'Operate', 'End']

    def __init__(self, mode):
        super().__init__()
        # Running mode of this gui. Options:
        #  - server-gui: Normal mode, starts bluesky server together with gui
        #  - client: starts only gui in client mode, can connect to existing
        #    server.
        self.mode = mode

        self.radarwidget = RadarWidget()
        # self.nd = ND(shareWidget=self.radarwidget)
        self.infowin = InfoWindow()
        self.settingswin = SettingsWindow()

        try:
            self.docwin = DocWindow(self)
        except Exception as e:
            print('Couldnt make docwindow:', e)
        # self.aman = AMANDisplay()
        gltimer = QTimer(self)
        gltimer.timeout.connect(self.radarwidget.update)
        # gltimer.timeout.connect(self.nd.updateGL)
        gltimer.start(50)

        gfxpath = bs.resource(bs.settings.gfx_path)

        if platform.system() == 'Darwin':
            app.instance().setWindowIcon(QIcon((gfxpath / 'bluesky.icns').as_posix()))
        else:
            app.instance().setWindowIcon(QIcon((gfxpath / 'icon.gif').as_posix()))

        uic.loadUi((gfxpath / 'mainwindow.ui').as_posix(), self)

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
                    self.hold :       ['hold.svg', 'Hold', self.buttonClicked],
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
                icon = QIcon((gfxpath / 'icons' / b[1][0]).as_posix())
                b[0].setIcon(icon)
            # Set tooltip
            if not b[1][1] is None:
                b[0].setToolTip(b[1][1])
            # Connect clicked signal
            b[0].clicked.connect(b[1][2])

        # If multiple scenario paths exist, add 'Open From' menu
        scenresource = bs.resource('scenario')
        if isinstance(scenresource, ResourcePath) and scenresource.nbases > 1:
            openfrom = QMenu('Open From', self.menuFile)
            self.menuFile.insertMenu(self.action_Save, openfrom)

            openpkg = openfrom.addAction('Package')
            openpkg.triggered.connect(lambda: self.show_file_dialog(scenresource.base(-1)))
            openusr = openfrom.addAction('User')
            openusr.triggered.connect(lambda: self.show_file_dialog(scenresource.base(0)))

        # Link menubar buttons
        self.action_Open.triggered.connect(self.show_file_dialog)
        self.action_Save.triggered.connect(self.buttonClicked)
        self.actionBlueSky_help.triggered.connect(self.show_doc_window)
        self.actionSettings.triggered.connect(self.settingswin.show)

        self.radarwidget.setParent(self.centralwidget)
        self.verticalLayout.insertWidget(0, self.radarwidget, 1)
        # Connect to io client's nodelist changed signal
        bs.net.nodes_changed.connect(self.nodesChanged)
        bs.net.actnodedata_changed.connect(self.actnodedataChanged)
        bs.net.event_received.connect(self.on_simevent_received)
        bs.net.stream_received.connect(self.on_simstream_received)
        bs.net.signal_quit.connect(self.closeEvent)

        self.nodetree.setVisible(False)
        self.nodetree.setIndentation(0)
        self.nodetree.setColumnCount(2)
        self.nodetree.setStyleSheet('padding:0px')
        self.nodetree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.nodetree.header().resizeSection(0, 130)
        self.nodetree.itemClicked.connect(self.nodetreeClicked)
        self.maxhostnum = 0
        self.hosts = dict()
        self.nodes = dict()
        self.actnode = ''

        fgcolor = '#%02x%02x%02x' % fg
        bgcolor = '#%02x%02x%02x' % bg

        self.stackText.setStyleSheet('color:' + fgcolor + '; background-color:' + bgcolor)
        self.lineEdit.setStyleSheet('color:' + fgcolor + '; background-color:' + bgcolor)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier \
                and event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right]:
            dlat = 1.0 / (self.radarwidget.zoom * self.radarwidget.ar)
            dlon = 1.0 / (self.radarwidget.zoom * self.radarwidget.flat_earth)
            if event.key() == Qt.Key.Key_Up:
                self.radarwidget.panzoom(pan=(dlat, 0.0))
            elif event.key() == Qt.Key.Key_Down:
                self.radarwidget.panzoom(pan=(-dlat, 0.0))
            elif event.key() == Qt.Key.Key_Left:
                self.radarwidget.panzoom(pan=(0.0, -dlon))
            elif event.key() == Qt.Key.Key_Right:
                self.radarwidget.panzoom(pan=(0.0, dlon))

        elif event.key() == Qt.Key.Key_Escape:
            self.closeEvent()

        elif event.key() == Qt.Key.Key_F11:  # F11 = Toggle Full Screen mode
            if not self.isFullScreen():
                self.showFullScreen()
            else:
                self.showNormal()

        else:
            # All other events go to the BlueSky console
            self.console.keyPressEvent(event)
        event.accept()

    def closeEvent(self, event=None):
        # Send quit to server if we own the host
        if self.mode != 'client':
            bs.net.send_event(b'QUIT')
        app.instance().closeAllWindows()
        # return True

    def actnodedataChanged(self, nodeid, nodedata, changed_elems):
        if nodeid != self.actnode:
            self.actnode = nodeid
            node = self.nodes[nodeid]
            self.nodelabel.setText(f'<b>Node</b> {node.host_num}:{node.node_num}')
            self.nodetree.setCurrentItem(node, 0, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def nodesChanged(self, data):
        for host_id, host_data in data.items():
            host = self.hosts.get(host_id)
            if not host:
                host = QTreeWidgetItem(self.nodetree)
                self.maxhostnum += 1
                host.host_num = self.maxhostnum
                host.host_id = host_id
                hostname = 'This computer' if host_id == bs.net.get_hostid() else str(host_id)
                f = host.font(0)
                f.setBold(True)
                host.setExpanded(True)
                btn = QPushButton(self.nodetree)
                btn.host_id = host_id
                btn.setText(hostname)
                btn.setFlat(True)
                btn.setStyleSheet('font-weight:bold')
                icon = bs.resource(bs.settings.gfx_path) / 'icons/addnode.svg'
                btn.setIcon(QIcon(icon.as_posix()))
                btn.setIconSize(QSize(24, 16))
                btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
                btn.setMaximumHeight(16)
                btn.clicked.connect(self.buttonClicked)
                self.nodetree.setItemWidget(host, 0, btn)
                self.hosts[host_id] = host

            for node_num, node_id in enumerate(host_data['nodes']):
                if node_id not in self.nodes:
                    # node_num = node_id[-2] * 256 + node_id[-1]
                    node = QTreeWidgetItem(host)
                    node.setText(0, f'{host.host_num}:{node_num + 1} <init>')
                    node.setText(1, '00:00:00')
                    node.node_id  = node_id
                    node.node_num = node_num + 1
                    node.host_num = host.host_num

                    self.nodes[node_id] = node

    def on_simevent_received(self, eventname, eventdata, sender_id):
        ''' Processing of events from simulation nodes. '''
        # ND window for selected aircraft
        if eventname == b'SHOWND':
            return
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
            speed, simdt, simt, simutc, ntraf, state, scenname = data
            simt = tim2txt(simt)[:-3]
            self.setNodeInfo(sender_id, simt, scenname)
            if sender_id == bs.net.actnode():
                acdata = bs.net.get_nodedata().acdata
                self.siminfoLabel.setText(u'<b>t:</b> %s, <b>\u0394t:</b> %.2f, <b>Speed:</b> %.1fx, <b>UTC:</b> %s, <b>Mode:</b> %s, <b>Aircraft:</b> %d, <b>Conflicts:</b> %d/%d, <b>LoS:</b> %d/%d'
                    % (simt, simdt, speed, simutc, self.modes[state], ntraf, acdata.nconf_cur, acdata.nconf_tot, acdata.nlos_cur, acdata.nlos_tot))

    def setNodeInfo(self, connid, time, scenname):
        node = self.nodes.get(connid)
        if node:
            node.setText(0, f'{node.host_num}:{node.node_num} <{scenname}>')
            node.setText(1, time)

    @pyqtSlot(QTreeWidgetItem, int)
    def nodetreeClicked(self, item, column):
        if item in self.hosts.values():
            item.setSelected(False)
            item.child(0).setSelected(True)
            bs.net.actnode(item.child(0).node_id)
        else:
            bs.net.actnode(item.node_id)


    @pyqtSlot()
    def buttonClicked(self):
        actdata = bs.net.get_nodedata()
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
            self.show_file_dialog()
        elif self.sender() == self.sameic:
            bs.net.send_event(b'STACK', 'IC IC')
        elif self.sender() == self.hold:
            bs.net.send_event(b'STACK', 'HOLD')
        elif self.sender() == self.op:
            bs.net.send_event(b'STACK', 'OP')
        elif self.sender() == self.fast:
            bs.net.send_event(b'STACK', 'FF')
        elif self.sender() == self.fast10:
            bs.net.send_event(b'STACK', 'FF 0:0:10')
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
            bs.net.send_event(b'STACK', 'SAVEIC')
        elif hasattr(self.sender(), 'host_id'):
            bs.net.send_event(b'ADDNODES', 1)

    def show_file_dialog(self, path=None):
        # Due to Qt5 bug in Windows, use temporarily Tkinter
        if platform.system().lower()=='windows':
            fname = fileopen()
        else:
            if path is None:
                path = bs.resource(bs.settings.scenario_path)

            if isinstance(path, ResourcePath):
                def getscenpath(resource):
                    # Find first path that contains scenario files
                    for p in resource.bases():
                        for f in p.glob('*.[Ss][Cc][Nn]'):
                            if f.name.lower() != 'ic.scn':
                                return p.as_posix()
                    return p.as_posix()
                scenpath = getscenpath(path)
            elif isinstance(path, Path):
                scenpath = path.as_posix()
            else:
                scenpath = path
            
            if platform.system().lower() == 'darwin':
                response = QFileDialog.getOpenFileName(self, 'Open file', scenpath, 'Scenario files (*.scn)')
            else:
                response = QFileDialog.getOpenFileName(self, 'Open file', scenpath, 'Scenario files (*.scn)', options=QFileDialog.Option.DontUseNativeDialog)
            fname = response[0] if isinstance(response, tuple) else response

        # Send IC command to stack with filename if selected, else do nothing
        if fname:
            bs.stack.stack('IC ' + str(fname))

    def show_doc_window(self, cmd=''):
        self.docwin.show_cmd_doc(cmd)
        self.docwin.show()
