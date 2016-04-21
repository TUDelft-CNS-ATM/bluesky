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
from ...sim.qtgl import PanZoomEvent, MainManager as manager


class Splash(QSplashScreen):
    """ Splash screen: BlueSky logo during start-up"""
    def __init__(self):
        super(Splash, self).__init__(QPixmap('data/graphics/splash.gif'), Qt.WindowStaysOnTopHint)


class MainWindow(QMainWindow):
    """ Qt window process: from .ui file read UI window-definitionof main window """

    def __init__(self, app, radarwidget):
        super(MainWindow, self).__init__()
        self.app = app
        uic.loadUi("./data/graphics/mainwindow.ui", self)

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

        for b in buttons.iteritems():
            # Set icon
            if not b[1][0] is None:
                icon = QIcon('data/graphics/icons/' + b[1][0])
                b[0].setIcon(icon)
            # Set tooltip
            if not b[1][1] is None:
                b[0].setToolTip(b[1][1])
            # Connect clicked signal
            b[0].clicked.connect(b[1][2])

        self.radarwidget = radarwidget
        radarwidget.setParent(self.centralwidget)
        self.verticalLayout.insertWidget(0, radarwidget, 1)
        # Connect to manager's nodelist changed signal
        manager.instance.nodes_changed.connect(self.nodesChanged)
        manager.instance.activenode_changed.connect(self.actnodeChanged)

        self.nodetree.setVisible(False)
        self.nodetree.setIndentation(0)
        self.nodetree.setColumnCount(2)
        self.nodetree.setStyleSheet('padding:0px')
        self.nodetree.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.nodetree.header().resizeSection(0, 130)
        self.nodetree.itemClicked.connect(self.nodetreeClicked)
        self.hosts = list()
        self.nodes = list()

    def closeEvent(self, event):
        self.app.quit()

    @pyqtSlot(int)
    def actnodeChanged(self, nodeid, connidx):
        self.nodelabel.setText('<b>Node</b> %d:%d' % nodeid)
        self.nodetree.setCurrentItem(self.hosts[nodeid[0]].child(nodeid[1]), 0, QItemSelectionModel.ClearAndSelect)

    @pyqtSlot(str, int)
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
            # host.setFont(0, f)
            # host.setText(0, hostname)
            host.setExpanded(True)
            btn = QPushButton(self.nodetree)
            # btn.setFont(0, f)
            btn.setText(hostname)
            btn.setFlat(True)
            btn.setStyleSheet('font-weight:bold')

            btn.setIcon(QIcon('data/graphics/icons/addnode.svg'))
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
            self.app.stack('IC IC')
        elif self.sender() == self.hold:
            self.app.stack('HOLD')
        elif self.sender() == self.op:
            self.app.stack('OP')
        elif self.sender() == self.fast:
            self.app.stack('FF')
        elif self.sender() == self.fast10:
            self.app.stack('FF 0:0:10')
        elif self.sender() == self.showac:
            self.radarwidget.show_traf = not self.radarwidget.show_traf
        elif self.sender() == self.showpz:
            self.radarwidget.show_pz = not self.radarwidget.show_pz
        elif self.sender() == self.showapt:
            self.radarwidget.show_apt = not self.radarwidget.show_apt
        elif self.sender() == self.showwpt:
            self.radarwidget.show_wpt = not self.radarwidget.show_wpt
        elif self.sender() == self.showlabels:
            self.radarwidget.show_lbl = not self.radarwidget.show_lbl
        elif self.sender() == self.showmap:
            self.radarwidget.show_map = not self.radarwidget.show_map
