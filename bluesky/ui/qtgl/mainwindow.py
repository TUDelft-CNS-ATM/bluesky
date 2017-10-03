""" Main window for the QTGL gui."""
import platform
import os
try:
    from PyQt5.QtCore import Qt, pyqtSlot, QItemSelectionModel, QSize
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import QMainWindow, QSplashScreen, QTreeWidgetItem, QPushButton
    from PyQt5 import uic
except ImportError:
    from PyQt4.QtCore import Qt, pyqtSlot, QSize
    from PyQt4.QtGui import QPixmap, QMainWindow, QIcon, QSplashScreen, \
        QItemSelectionModel, QTreeWidgetItem, QPushButton
    from PyQt4 import uic

# Local imports
from bluesky.simulation.qtgl import StackTextEvent, PanZoomEvent, MainManager as manager
from bluesky import settings


is_osx = platform.system() == 'Darwin'

# Register settings defaults
settings.set_variable_defaults(gfx_path='data/graphics', stack_text_color=(0, 255, 0), stack_background_color=(102, 102, 102))

fg = settings.stack_text_color
bg = settings.stack_background_color

class Splash(QSplashScreen):
    """ Splash screen: BlueSky logo during start-up"""
    def __init__(self):
        super(Splash, self).__init__(QPixmap(os.path.join(settings.gfx_path, 'splash.gif')), Qt.WindowStaysOnTopHint)


class MainWindow(QMainWindow):
    """ Qt window process: from .ui file read UI window-definitionof main window """

    def __init__(self, app, radarwidget):
        super(MainWindow, self).__init__()
        self.app = app
        if is_osx:
            self.app.setWindowIcon(QIcon(os.path.join(settings.gfx_path, 'bluesky.icns')))
        else:
            self.app.setWindowIcon(QIcon(os.path.join(settings.gfx_path, 'icon.gif')))

        uic.loadUi(os.path.join(settings.gfx_path, 'mainwindow.ui'), self)

        # list of buttons to connect to, give icons, and tooltips
        #           the button         the icon      the tooltip    the callback
        buttons = { self.zoomin  :    ['zoomin.svg', 'Zoom in', self.buttonClicked],
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
        self.action_Open.triggered.connect(app.show_file_dialog)
        self.action_Save.triggered.connect(self.buttonClicked)
        self.actionBlueSky_help.triggered.connect(app.show_doc_window)

        self.radarwidget = radarwidget
        radarwidget.setParent(self.centralwidget)
        self.verticalLayout.insertWidget(0, radarwidget, 1)
        # Connect to manager's nodelist changed signal
        manager.instance.nodes_changed.connect(self.nodesChanged)
        manager.instance.activenode_changed.connect(self.actnodeChanged)
        # Connect widgets with each other
        self.console.cmdline_stacked.connect(self.radarwidget.cmdline_stacked)

        self.nodetree.setVisible(False)
        self.nodetree.setIndentation(0)
        self.nodetree.setColumnCount(2)
        self.nodetree.setStyleSheet('padding:0px')
        self.nodetree.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.nodetree.header().resizeSection(0, 130)
        self.nodetree.itemClicked.connect(self.nodetreeClicked)
        self.hosts = list()
        self.nodes = list()

        fgcolor = '#%02x%02x%02x' % fg
        bgcolor = '#%02x%02x%02x' % bg

        self.stackText.setStyleSheet('color:' + fgcolor + '; background-color:' + bgcolor)
        self.lineEdit.setStyleSheet('color:' + fgcolor + '; background-color:' + bgcolor)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier \
                and event.key() in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            dlat = 1.0 / (self.radarwidget.zoom * self.radarwidget.ar)
            dlon = 1.0 / (self.radarwidget.zoom * self.radarwidget.flat_earth)
            if event.key() == Qt.Key_Up:
                self.radarwidget.event(PanZoomEvent(pan=(dlat, 0.0)))
            elif event.key() == Qt.Key_Down:
                self.radarwidget.event(PanZoomEvent(pan=(-dlat, 0.0)))
            elif event.key() == Qt.Key_Left:
                self.radarwidget.event(PanZoomEvent(pan=(0.0, -dlon)))
            elif event.key() == Qt.Key_Right:
                self.radarwidget.event(PanZoomEvent(pan=(0.0, dlon)))

        elif event.key() == Qt.Key_Escape:
            self.app.quit()

        elif event.key() == Qt.Key_F11:  # F11 = Toggle Full Screen mode
            if not self.isFullScreen():
                self.showFullScreen()
            else:
                self.showNormal()

        else:
            # All other events go to the BlueSky console
            self.console.keyPressEvent(event)

    def closeEvent(self, event):
        self.app.quit()

    @pyqtSlot(tuple, int)
    def actnodeChanged(self, nodeid, connidx):
        self.nodelabel.setText('<b>Node</b> %d:%d' % nodeid)
        self.nodetree.setCurrentItem(self.hosts[nodeid[0]].child(nodeid[1]), 0, QItemSelectionModel.ClearAndSelect)

    @pyqtSlot(str, tuple, int)
    def nodesChanged(self, address, nodeid, connidx):
        if nodeid[0] < len(self.hosts):
            host = self.hosts[nodeid[0]]
        else:
            host = QTreeWidgetItem(self.nodetree)
            hostname = address
            if address in ['127.0.0.1', 'localhost']:
                hostname = 'This computer'
            f = host.font(0)
            f.setBold(True)
            host.setExpanded(True)
            btn = QPushButton(self.nodetree)
            btn.setText(hostname)
            btn.setFlat(True)
            btn.setStyleSheet('font-weight:bold')

            btn.setIcon(QIcon(os.path.join(settings.gfx_path, 'icons/addnode.svg')))
            btn.setIconSize(QSize(24, 16))
            btn.setLayoutDirection(Qt.RightToLeft)
            btn.setMaximumHeight(16)
            btn.clicked.connect(manager.instance.addNode)
            self.nodetree.setItemWidget(host, 0, btn)
            self.hosts.append(host)

        node = QTreeWidgetItem(host)
        node.setText(0, '%d:%d <init>' % nodeid)
        node.setText(1, '00:00:00')
        node.connidx = connidx
        node.nodeid  = nodeid
        self.nodes.append(node)

    def setNodeInfo(self, connidx, time, scenname):
        node = self.nodes[connidx]
        node.setText(0, '%d:%d <'  % node.nodeid + scenname + '>')
        node.setText(1, time)

    @pyqtSlot(QTreeWidgetItem, int)
    def nodetreeClicked(self, item, column):
        if item in self.hosts:
            item.setSelected(False)
            item.child(0).setSelected(True)
            connidx = item.child(0).connidx
        else:
            connidx = item.connidx
        manager.instance.setActiveNode(connidx)

    @pyqtSlot()
    def buttonClicked(self):
        if self.sender() == self.shownodes:
            vis = not self.nodetree.isVisible()
            self.nodetree.setVisible(vis)
            self.shownodes.setText('>' if vis else '<')
        if self.sender() == self.zoomin:
            self.app.notify(self.app, PanZoomEvent(zoom=1.4142135623730951))
        elif self.sender() == self.zoomout:
            self.app.notify(self.app, PanZoomEvent(zoom=0.70710678118654746))
        elif self.sender() == self.pandown:
            self.app.notify(self.app, PanZoomEvent(pan=(-0.5,  0.0)))
        elif self.sender() == self.panup:
            self.app.notify(self.app, PanZoomEvent(pan=( 0.5,  0.0)))
        elif self.sender() == self.panleft:
            self.app.notify(self.app, PanZoomEvent(pan=( 0.0, -0.5)))
        elif self.sender() == self.panright:
            self.app.notify(self.app, PanZoomEvent(pan=( 0.0,  0.5)))
        elif self.sender() == self.ic:
            self.app.show_file_dialog()
        elif self.sender() == self.sameic:
            manager.sendEvent(StackTextEvent(cmdtext='IC IC'))
        elif self.sender() == self.hold:
            manager.sendEvent(StackTextEvent(cmdtext='HOLD'))
        elif self.sender() == self.op:
            manager.sendEvent(StackTextEvent(cmdtext='OP'))
        elif self.sender() == self.fast:
            manager.sendEvent(StackTextEvent(cmdtext='FF'))
        elif self.sender() == self.fast10:
            manager.sendEvent(StackTextEvent(cmdtext='FF 0:0:10'))
        elif self.sender() == self.showac:
            self.radarwidget.show_traf = not self.radarwidget.show_traf
        elif self.sender() == self.showpz:
            self.radarwidget.show_pz = not self.radarwidget.show_pz
        elif self.sender() == self.showapt:
            if self.radarwidget.show_apt < 3:
                self.radarwidget.show_apt += 1
            else:
                self.radarwidget.show_apt = 0
        elif self.sender() == self.showwpt:
            if self.radarwidget.show_wpt < 2:
                self.radarwidget.show_wpt += 1
            else:
                self.radarwidget.show_wpt = 0
        elif self.sender() == self.showlabels:
            self.radarwidget.show_lbl -= 1
            if self.radarwidget.show_lbl < 0:
                self.radarwidget.show_lbl = 2
        elif self.sender() == self.showmap:
            self.radarwidget.show_map = not self.radarwidget.show_map
        elif self.sender() == self.action_Save:
            manager.sendEvent(StackTextEvent(cmdtext='SAVEIC'))
