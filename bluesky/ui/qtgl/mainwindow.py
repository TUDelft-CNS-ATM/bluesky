""" Main window for the QTGL gui."""
from pathlib import Path
import platform

from PyQt6.QtWidgets import QApplication as app, QWidget, QMainWindow, \
    QSplashScreen, QTreeWidgetItem, QPushButton, QFileDialog, QDialog, \
    QTreeWidget, QVBoxLayout, QDialogButtonBox, QMenu, QLabel
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QItemSelectionModel, QSize, QEvent, pyqtProperty
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6 import uic

# Local imports
import bluesky as bs
from bluesky.core import Base, Signal
from bluesky.network.discovery import Discovery

from bluesky import stack
from bluesky.stack.argparser import PosArg
from bluesky.pathfinder import ResourcePath
from bluesky.tools.misc import tim2txt
from bluesky.network import subscriber, context as ctx
from bluesky.network.common import get_ownip, seqidx2id, seqid2idx
import bluesky.network.sharedstate as ss

from bluesky.ui import palette

# Child windows
from bluesky.ui.qtgl.docwindow import DocWindow
from bluesky.ui.qtgl.infowindow import InfoWindow
from bluesky.ui.qtgl.settingswindow import SettingsWindow
# from bluesky.ui.qtgl.nd import ND

if platform.system().lower() == "windows":
    from bluesky.ui.pygame.dialog import fileopen
    import winreg
    def isdark():
        ''' Returns true if app is in dark mode, false otherwise. '''
        try:
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(registry, r"Software\\Microsoft\\\Windows\\CurrentVersion\\Themes\\Personalize")
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return value == 0
        except FileNotFoundError:
            return False
else:
    def isdark():
        ''' Returns true if app is in dark mode, false otherwise. '''
        p = app.instance().style().standardPalette()
        return (p.color(p.ColorRole.Window).value() < p.color(p.ColorRole.WindowText).value())


# Register settings defaults
bs.settings.set_variable_defaults(gfx_path='graphics', start_location='EHAM')

palette.set_default_colours(stack_text=(0, 255, 0),
                            stack_background=(102, 102, 102))


class Splash(QSplashScreen):
    """ Splash screen: BlueSky logo during start-up"""
    def __init__(self):
        splashfile = bs.resource(bs.settings.gfx_path) / 'splash.gif'
        super().__init__(QPixmap(splashfile.as_posix()), Qt.WindowType.WindowStaysOnTopHint)


class DiscoveryDialog(QDialog):
    def __init__(self, comm_id, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setMinimumSize(200,200) # To prevent Geometry error
        self.servers = []
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

        self.discovery = Discovery(comm_id)

        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self.discovery.update)
        self.discovery_timer.start(1000)
        self.discovery.server_discovered.connect(self.add_srv)
        self.discovery.start()

    def add_srv(self, address, ports):
        for server in self.servers:
            if address == server.address and ports == server.ports:
                # We already know this server, skip
                return
        server = QTreeWidgetItem(self.serverview)
        server.address = address
        server.ports = ports
        server.hostname = 'This computer' if address == get_ownip() else address
        server.setText(0, server.hostname)

        server.setText(1, '{},{}'.format(*ports))
        self.servers.append(server)

    def on_accept(self):
        server = self.serverview.currentItem()
        if server:
            self.discovery_timer.stop()
            self.discovery.stop()
            hostname = server.address
            rport, sport = server.ports
            bs.net.connect(hostname=hostname, recv_port=rport, send_port=sport)
            self.close()


class MainWindow(QMainWindow, Base):
    """ Qt window process: from .ui file read UI window-definition of main window """

    modes = ['Init', 'Hold', 'Operate', 'End']

    # Per remote node attributes
    nconf_cur: ss.ActData[int] = ss.ActData(0, group='acdata')
    nconf_tot: ss.ActData[int] = ss.ActData(0, group='acdata')
    nlos_cur: ss.ActData[int] = ss.ActData(0, group='acdata')
    nlos_tot: ss.ActData[int] = ss.ActData(0, group='acdata')

    show_map: ss.ActData[bool] = ss.ActData(True)
    show_coast: ss.ActData[bool] = ss.ActData(True)
    show_wpt: ss.ActData[int] = ss.ActData(1)
    show_apt: ss.ActData[int] = ss.ActData(1)
    show_pz: ss.ActData[bool] = ss.ActData(False)
    show_traf: ss.ActData[bool] = ss.ActData(True)
    show_lbl: ss.ActData[int] = ss.ActData(2)

    @pyqtProperty(str)
    def style(self):
        ''' Returns "dark"" if app is in dark mode, "light" otherwise. '''
        return "dark" if self.darkmode else "light"

    def __init__(self, mode):
        super().__init__()
        # Running mode of this gui. Options:
        #  - server-gui: Normal mode, starts bluesky server together with gui
        #  - client: starts only gui in client mode, can connect to existing
        #    server.
        self.mode = mode
        self.running = True

        # self.nd = ND(shareWidget=self.radarwidget)
        self.infowin = InfoWindow()
        self.settingswin = SettingsWindow()
        self.darkmode = isdark()

        try:
            self.docwin = DocWindow(self)
        except Exception as e:
            print('Couldnt make docwindow:', e)
        # self.aman = AMANDisplay()
        

        gfxpath = bs.resource(bs.settings.gfx_path)

        if platform.system() == 'Darwin':
            app.instance().setWindowIcon(QIcon((gfxpath / 'bluesky.icns').as_posix()))
        else:
            app.instance().setWindowIcon(QIcon((gfxpath / 'icon.gif').as_posix()))

        uic.loadUi((gfxpath / 'mainwindow.ui').as_posix(), self)
        gltimer = QTimer(self)
        gltimer.timeout.connect(self.radarwidget.update)
        # gltimer.timeout.connect(self.nd.updateGL)
        gltimer.start(50)

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

        # Connect to io client's nodelist changed signal
        bs.net.node_added.connect(self.nodesChanged)
        bs.net.server_added.connect(self.serversChanged)

        # Tell BlueSky that this is the screen object for this client
        bs.scr = self

        # Signals we want to emit
        self.panzoom_event = Signal('state-changed.panzoom')

        # Set position default from settings
        lat, lon, _ = PosArg().parse(bs.settings.start_location)
        ss.setdefault('pan', [lat, lon], group='panzoom')


        # self.nodetree.setVisible(False)
        self.nodetree.setIndentation(0)
        self.nodetree.setColumnCount(2)
        self.nodetree.setStyleSheet('padding:0px')
        self.nodetree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.nodetree.header().resizeSection(0, 130)
        self.nodetree.itemClicked.connect(self.nodetreeClicked)
        self.maxservnum = 0
        self.servers = dict()
        self.nodes = dict()
        self.actnode = ''

        self.splitter.setSizes([1, 0])
        self.splitter_2.setSizes([1, 0])
        self.setStyleSheet()

        # Remove keyboard focus from all children of self.databox
        # (not possible to do in QDesigner)
        def recursiveNoFocus(w):
            for child in w.findChildren(QWidget):
                recursiveNoFocus(child)
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        recursiveNoFocus(self.databox)

    def setStyleSheet(self, contents=''):
        if not contents:
            with open(bs.resource(bs.settings.gfx_path) / 'bluesky.qss') as style:
                super().setStyleSheet(style.read())

    @stack.command
    def swrad(self, switch: 'txt', arg: int|None = None):
        ''' Switch on/off elements and background of map/radar view 
        
            Usage:
                SWRAD GEO/GRID/APT/VOR/WPT/LABEL/ADSBCOVERAGE/TRAIL/POLY [dt]/[value]
        '''
        match switch:
            case 'GEO':
                self.show_coast = not self.show_coast
            case 'SAT':
                self.show_map = not self.show_map
            case 'APT':
                if arg is not None:
                    self.show_apt = min(2,max(0,arg))
                else:
                    self.show_apt = (self.show_apt + 1) % 3
            case 'WPT':
                if arg is not None:
                    self.show_wpt = min(2,max(0,arg))
                else:
                    self.show_wpt = (self.show_wpt + 1) % 3
            case 'LABEL':
                if arg is not None:
                    self.show_lbl = min(2,max(0,arg))
                else:
                    self.show_lbl = (self.show_lbl + 1) % 3
            case 'SYM':
                if arg is None:
                    arg = 0 if self.show_pz else (2 if self.show_traf else 1)
                self.show_traf = arg > 0
                self.show_pz = arg > 1

    @stack.command
    def mcre(self, args: 'string'):
        """ Create one or more random aircraft in a specified area 
        
            When called from the client (gui), MCRE will use the current screen bounds as area to create aircraft in.
            When called from a scenario, the simulation reference area will be used.
        """
        if not args:
            return stack.forward('MCRE')

        stack.forward(f'INSIDE {" ".join(str(el) for el in bs.ref.area.bbox)} MCRE {args}')

    @stack.command(annotations='pandir/latlon', brief='PAN latlon/acid/airport/waypoint/LEFT/RIGHT/UP/DOWN')
    def pan(self, *args):
        "Pan screen (move view) to a waypoint, direction or aircraft"
        store = ss.get(group='panzoom')
        store.pan = list(args)
        self.panzoom_event.emit(store)
        return True

    @stack.commandgroup(annotations='float/txt', brief='ZOOM IN/OUT/factor')
    def zoom(self, factor: float):
        ''' ZOOM: Zoom in and out in the radar view. 
        
            Arguments:
            - factor: IN/OUT to zoom in/out by a factor sqrt(2), or
                      'factor' to set zoom to specific value.
        '''
        store = ss.get(group='panzoom')
        store.zoom = factor
        self.panzoom_event.emit(store)
        return True

    @zoom.subcommand(name='IN')
    def zoomin(self, factor:float|None=None):
        ''' ZOOM IN: change zoom level up, relative to previous value '''
        store = ss.get(group='panzoom')
        store.zoom *= (factor or 1.4142135623730951)
        self.panzoom_event.emit(store)
        return True

    @zoom.subcommand(name='OUT')
    def zoomout(self, factor:float|None=None):
        ''' ZOOM OUT: change zoom level down, relative to previous value '''
        store = ss.get(group='panzoom')
        store.zoom *= (factor or 0.7071067811865475)
        self.panzoom_event.emit(store)
        return True

    def getviewctr(self):
        return self.radarwidget.pan

    def getviewbounds(self): # Return current viewing area in lat, lon
        return self.radarwidget.viewportlatlon()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier \
                and event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right]:
            dlat = 1.0 / (self.radarwidget.zoom * self.radarwidget.ar)
            dlon = 1.0 / (self.radarwidget.zoom * self.radarwidget.flat_earth)
            if event.key() == Qt.Key.Key_Up:
                self.radarwidget.setpanzoom(pan=(dlat, 0.0), absolute=False)
            elif event.key() == Qt.Key.Key_Down:
                self.radarwidget.setpanzoom(pan=(-dlat, 0.0), absolute=False)
            elif event.key() == Qt.Key.Key_Left:
                self.radarwidget.setpanzoom(pan=(0.0, -dlon), absolute=False)
            elif event.key() == Qt.Key.Key_Right:
                self.radarwidget.setpanzoom(pan=(0.0, dlon), absolute=False)

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

    @stack.command(name='QUIT', annotations='', aliases=('CLOSE', 'END', 'EXIT', 'Q', 'STOP'))
    def closeEvent(self, event=None):
        if self.running:
            self.running = False
            # Send quit to server if we own it
            if self.mode != 'client':
                bs.net.send(b'QUIT', to_group=bs.server.server_id)
            app.instance().closeAllWindows()
            # return True

    @subscriber
    def echo(self, text, flags=None, sender_id=None):
        refnode = sender_id or ctx.sender_id or bs.net.act_id
        # Always update the store
        store = ss.get(refnode)
        store.echotext.append(text)
        store.echoflags.append(flags)
        # Directly echo if message corresponds to active node
        if refnode == bs.net.act_id:
            return self.console.echo(text, flags)

    def changeEvent(self, event: QEvent):
        # Detect dark/light mode switch
        if event.type() == event.Type.PaletteChange and self.darkmode != isdark():
            self.darkmode = isdark()
            self.setStyleSheet()

        return super().changeEvent(event)


    def actnodedataChanged(self, nodeid, nodedata, changed_elems):
        if nodeid != self.actnode:
            self.actnode = nodeid
            node = self.nodes[nodeid]
            self.nodelabel.setText(f'<b>Node</b> {node.serv_num}:{node.node_num}')
            self.nodetree.setCurrentItem(node, 0, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def serversChanged(self, server_id):
        server = self.servers.get(server_id)
        if not server:
            server = QTreeWidgetItem(self.nodetree)
            self.maxservnum += 1
            server.serv_num = self.maxservnum
            server.server_id = server_id
            hostname = 'Ungrouped' if server_id == b'0' else 'This computer'
            f = server.font(0)
            f.setBold(True)
            server.setExpanded(True)
            if server_id != b'0':
                btn = QPushButton(self.nodetree)
                btn.server_id = server_id
                btn.setText(hostname)
                btn.setFlat(True)
                btn.setStyleSheet('font-weight:bold')
                icon = bs.resource(bs.settings.gfx_path) / 'icons/addnode.svg'
                btn.setIcon(QIcon(icon.as_posix()))
                btn.setIconSize(QSize(40 if server_id == b'0' else 24, 16))
                btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
                btn.setMaximumHeight(16)
                btn.clicked.connect(self.buttonClicked)
                self.nodetree.setItemWidget(server, 0, btn)

                # Move nodes from ungrouped if they belong to this server
                ungrouped: QTreeWidgetItem = self.servers.get(b'0')
                ucount = 0
                if ungrouped:
                    for node in ungrouped.takeChildren():
                        if node.node_id[:-1] + seqidx2id(0) == server_id:
                            server.addChild(node)
                        else:
                            ungrouped.addChild(node)
                            ucount += 1
                    if not ucount:
                        ungrouped.setHidden(True)
            else:
                self.nodetree.setItemWidget(server, 0, QLabel(hostname, parent=self.nodetree))
            self.servers[server_id] = server


    def nodesChanged(self, node_id):
        if node_id not in self.nodes:
            #print(node_id, 'added to list')
            server_id = node_id[:-1] + seqidx2id(0)
            if server_id not in bs.net.servers:
                server_id = b'0'
            if server_id not in self.servers:
                self.serversChanged(server_id)
            server = self.servers.get(server_id)
            server.setHidden(False)
            node_num = seqid2idx(node_id[-1])
            node = QTreeWidgetItem(server)
            node.setText(0, f'{server.serv_num}:{node_num} <init>')
            node.setText(1, '00:00:00')
            node.node_id  = node_id
            node.node_num = node_num
            node.serv_num = server.serv_num

            self.nodes[node_id] = node

    @subscriber(topic='SHOWDIALOG')
    def on_showdialog_received(self, dialog, args=''):
        ''' Processing of events from simulation nodes. '''
        # dialog = data.get('dialog')
        # args   = data.get('args')
        if dialog == 'OPENFILE':
            self.show_file_dialog()
        elif dialog == 'DOC':
            self.show_doc_window(args)

    @subscriber(topic='SIMINFO')
    def on_siminfo_received(self, speed, simdt, simt, simutc, ntraf, state, scenname):
        simt = tim2txt(simt)[:-3]
        self.setNodeInfo(ctx.sender_id, simt, scenname)
        if ctx.sender_id == bs.net.act_id:
            self.siminfoLabel.setText(u'<b>t:</b> %s, <b>\u0394t:</b> %.2f, <b>Speed:</b> %.1fx, <b>UTC:</b> %s, <b>Mode:</b> %s, <b>Aircraft:</b> %d, <b>Conflicts:</b> %d/%d, <b>LoS:</b> %d/%d'
                % (simt, simdt, speed, simutc, self.modes[state], ntraf, self.nconf_cur, self.nconf_tot, self.nlos_cur, self.nlos_tot))

    def setNodeInfo(self, connid, time, scenname):
        node = self.nodes.get(connid)
        if node:
            node.setText(0, f'{node.serv_num}:{node.node_num} <{scenname}>')
            node.setText(1, time)

    @pyqtSlot(QTreeWidgetItem, int)
    def nodetreeClicked(self, item, column):
        if item in self.servers.values():
            item.setSelected(False)
            item.child(0).setSelected(True)
            bs.net.actnode(item.child(0).node_id)
        else:
            bs.net.actnode(item.node_id)


    @pyqtSlot()
    def buttonClicked(self):
        if self.sender() == self.zoomin:
            self.radarwidget.setpanzoom(zoom=1.4142135623730951, absolute=False)
        elif self.sender() == self.zoomout:
            self.radarwidget.setpanzoom(zoom=0.70710678118654746, absolute=False)
        elif self.sender() == self.pandown:
            self.radarwidget.setpanzoom(pan=(-0.5,  0.0), absolute=False)
        elif self.sender() == self.panup:
            self.radarwidget.setpanzoom(pan=( 0.5,  0.0), absolute=False)
        elif self.sender() == self.panleft:
            self.radarwidget.setpanzoom(pan=( 0.0, -0.5), absolute=False)
        elif self.sender() == self.panright:
            self.radarwidget.setpanzoom(pan=( 0.0,  0.5), absolute=False)
        elif self.sender() == self.ic:
            self.show_file_dialog()
        elif self.sender() == self.sameic:
            stack.stack('IC IC')
        elif self.sender() == self.hold:
            stack.stack('HOLD')
        elif self.sender() == self.op:
            stack.stack('OP')
        elif self.sender() == self.fast:
            stack.stack('FF')
        elif self.sender() == self.fast10:
            stack.stack('FF 0:0:10')
        elif self.sender() == self.showac:
            stack.stack('SHOWTRAF')
        elif self.sender() == self.showpz:
            stack.stack('SHOWPZ')
        elif self.sender() == self.showapt:
            stack.stack('SHOWAPT')
        elif self.sender() == self.showwpt:
            stack.stack('SHOWWPT')
        elif self.sender() == self.showlabels:
            stack.stack('LABEL')
        elif self.sender() == self.showmap:
            stack.stack('SHOWMAP')
        elif self.sender() == self.action_Save:
            stack.stack('SAVEIC')
        elif hasattr(self.sender(), 'server_id'):
            bs.net.send(b'ADDNODES', 1, self.sender().server_id)

    def show_file_dialog(self, path=None):
        # Due to Qt5 bug in Windows, use temporarily Tkinter
        if platform.system().lower()=='windows':
            fname = fileopen()
        else:
            if path is None or isinstance(path, bool):
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
