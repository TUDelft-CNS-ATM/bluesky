from os import path
from collections import defaultdict
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QVBoxLayout, QScrollArea, QGroupBox, QWidget, \
    QFormLayout, QLabel, QSpinBox, QCheckBox, QLineEdit, QHBoxLayout, \
        QTreeWidget, QTreeWidgetItem, QFrame, QPushButton, QLayout, QComboBox, \
        QListWidget, QListWidgetItem

import bluesky as bs
from bluesky.network.common import seqidx2id
from bluesky.network.subscriber import subscribe
import bluesky.network.sharedstate as ss
from bluesky.core import Base


def sel_palette(value, changed_fun):
    wid = QComboBox()

    palfiles = [path.basename(f) for f in (bs.resource(bs.settings.gfx_path) / 'palettes').glob('*')]
    wid.addItems(palfiles)
    wid.setCurrentText(value)
    wid.currentTextChanged.connect(changed_fun)
    return wid

def sel_perf(value, changed_fun):
    wid = QComboBox()
    wid.addItems(['openap', 'bada', 'legacy'])
    wid.setCurrentText(value)
    wid.currentTextChanged.connect(changed_fun)
    return wid


class PluginCheckList(QListWidget):
    def __init__(self, value, changed_fun, avail_plugins):
        super().__init__()
        self.changed_fun = changed_fun
        self.curvalue = {v.upper() for v in value}
        if avail_plugins:
            for name in avail_plugins:
                row = QListWidgetItem(name)
                row.name = name
                row.setFlags(row.flags()|Qt.ItemIsUserCheckable)
                row.setCheckState(Qt.Checked if name in self.curvalue else Qt.Unchecked)
                self.addItem(row)
        self.itemChanged.connect(self.onitemchanged)

    def onitemchanged(self, item):
        if item.checkState() & Qt.Checked:
            self.curvalue.add(item.name)
        else:
            self.curvalue -= {item.name}
        self.changed_fun(list(self.curvalue))

class SettingsWindow(QWidget, Base):
    customwids = {
        'colour_palette': sel_palette,
        'performance_model': sel_perf,
        'enabled_plugins': PluginCheckList
    }
    def __init__(self):
        super().__init__()
        self.resize(600, 500)
        self.populated = False
        self.maxservnum = 0
        self.servers = dict()
        self.changed = defaultdict(dict)
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(1, 1, 1, 1)
        self.scrollarea = QScrollArea()
        self.scrollarea.layout = QVBoxLayout()
        self.layout().addWidget(self.scrollarea)
        bottombar = QWidget()
        bottombar.setLayout(QHBoxLayout())
        bottombar.layout().setContentsMargins(11, 1, 1, 1)
        self.changedlabel = QLabel()
        self.resetbtn = QPushButton('Reset')
        self.resetbtn.setFixedWidth(100)
        self.resetbtn.setEnabled(False)
        self.resetbtn.clicked.connect(self.btnclicked)
        self.savebtn = QPushButton('Save')
        self.savebtn.setFixedWidth(100)
        self.savebtn.setEnabled(False)
        self.savebtn.clicked.connect(self.btnclicked)

        bottombar.layout().addWidget(self.changedlabel)
        bottombar.layout().addWidget(self.resetbtn)
        bottombar.layout().addWidget(self.savebtn)
        self.layout().addWidget(bottombar)

        self.nodetree = QTreeWidget()
        self.nodetree.setFixedWidth(200)
        self.nodetree.setIndentation(0)
        self.nodetree.setColumnCount(2)
        self.nodetree.setStyleSheet('padding:0px')
        self.nodetree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.nodetree.header().resizeSection(0, 130)
        self.nodetree.setHeaderHidden(True)
        self.nodetree.itemClicked.connect(self.nodetreeClicked)
        self.nodesettings = QWidget()
        self.nodesettings.setLayout(QVBoxLayout())
        container = QWidget()
        container.setLayout(self.scrollarea.layout)
        self.scrollarea.layout.setAlignment(Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        self.scrollarea.setWidget(container)
        self.scrollarea.setWidgetResizable(True)

        bs.net.node_added.connect(self.nodesChanged)
        # Subscribe to simulation settings SharedState
        subscribe('SIMSETTINGS')

    def show(self):
        if not self.populated:
            self.populate()
        super().show()

    def populate(self):
        top = bs.settings._settings_hierarchy['bluesky']
        netset = top['network']
        netitems = {**netset['discovery'], **netset['server']}
        net = self.make_settings_box('Network settings', netitems)
        self.scrollarea.layout.addWidget(net)

        pathitems = {n:v for n, v in bs.settings.__dict__.items() if 'path' in n}
        paths = self.make_settings_box('Paths', pathitems)
        self.scrollarea.layout.addWidget(paths)

        guiitems = {n:v for n, v in top['ui']['qtgl']['radarwidget'].items() if 'path' not in n}
        guiitems['colour_palette'] = bs.settings.colour_palette
        gui = self.make_settings_box('Gui settings', guiitems, target='gui')
        self.scrollarea.layout.addWidget(gui)

        sim = QGroupBox('Simulation settings')
        sim.setLayout(QHBoxLayout())
        self.scrollarea.layout.addWidget(sim)
        sim.layout().addWidget(self.nodetree)
        sim.layout().addWidget(self.nodesettings)


    def nodesChanged(self, node_id):
        server_id = node_id[:-1] + seqidx2id(0)
        server = self.servers.get(server_id)
        if not server:
            server = QTreeWidgetItem(self.nodetree)
            self.maxservnum += 1
            server.serv_num = self.maxservnum
            server.serv_id = server_id
            server.nodes = [i for i in bs.net.nodes if i.startswith(server_id[:-1])]
            hostname = 'This computer'# if serv_id == bs.net.get_hostid() else str(serv_id)
            f = server.font(0)
            f.setBold(True)
            server.setExpanded(True)
            server.setText(0, hostname)
            server.setText(1, f'(nodes: {len(server.nodes)})')
            self.servers[server_id] = server

    @pyqtSlot(QTreeWidgetItem, int)
    def nodetreeClicked(self, item, column):
        if item in self.servers.values():
            simsettings = dict()
            plugins = list()

            for node in item.nodes:
                store = ss.get(node, 'simsettings')
                simsettings.update(store.settings)
                plugins = store.plugins

            # First clear any old items in the node settings layout
            clear_layout(self.nodesettings.layout())

            top = simsettings['bluesky']
            traf = self.make_settings_box('Traffic', top['traffic'], target=item.serv_id)
            stack = self.make_settings_box(
                'Stack', top['stack'], target=item.serv_id)
            sim = self.make_settings_box(
                'Simulation', top['simulation'], maxdepth=0, target=item.serv_id)

            misc = self.make_settings_box('Misc', top['tools'], maxdepth=0, target=item.serv_id, avail_plugins=plugins)
            self.nodesettings.layout().addWidget(traf)
            self.nodesettings.layout().addWidget(stack)
            self.nodesettings.layout().addWidget(sim)
            self.nodesettings.layout().addWidget(misc)
            for name, plugin in simsettings.items():
                if name == 'bluesky':
                    continue
                pbox = self.make_settings_box(
                    name.capitalize(), plugin, target=item.serv_id)
                self.nodesettings.layout().addWidget(pbox)

    def add_row(self, box, name, value, depth=0, maxdepth=1, target='common', **kwargs):
        if isinstance(value, dict):
            if depth < maxdepth:
                line = QFrame()
                
                line.setFrameShape(QFrame.Shape.HLine)
                box.layout().addRow(QLabel(f'<b>{name}</b>'), line)
            for cname, cvalue in value.items():
                self.add_row(box, cname, cvalue, depth=depth+1, target=target, **kwargs)
            return
        if name in SettingsWindow.customwids:
            wid = SettingsWindow.customwids[name](
                value, self.input_changed, **kwargs)
        elif isinstance(value, bool):
            wid = QCheckBox('')
            wid.setChecked(value)
            wid.clicked.connect(self.input_changed)
        elif isinstance(value, int):
            wid = QSpinBox()
            wid.setMaximum(100000)
            wid.setValue(value)
            wid.setFixedWidth(300)
            wid.valueChanged.connect(self.input_changed)
        else:
            wid = QLineEdit(str(value))
            wid.setFixedWidth(300)
            wid.textEdited.connect(self.input_changed)
        wid.origvalue = value
        wid.name = name
        wid.target = target
        box.layout().addRow(QLabel(name), wid)

    def input_changed(self, value):
        wid = self.sender()
        # Store the changed value in a dict of all changed values
        self.changed[wid.target][wid.name] = value
        self.changedlabel.setText('Save the changes and restart BlueSky for the changes to take effect.')
        self.resetbtn.setEnabled(True)
        self.savebtn.setEnabled(True)

    def make_settings_box(self, name, items, maxdepth=1, target='common', **kwargs):
        box = QGroupBox(name)
        box.setLayout(QFormLayout())
        for rowname, value in items.items():
            self.add_row(box, rowname, value, maxdepth=maxdepth,
                         target=target, **kwargs)

        return box

    def reset(self, branch=None):
        if hasattr(branch, 'layout'):
            layout = getattr(branch, 'layout')
            layout = layout() if callable(layout) else layout
            if layout is not branch and layout is not None:
                self.reset(layout)
                return
        if isinstance(branch, QLayout):
            for i in range(branch.count()):
                self.reset(branch.itemAt(i).widget())
        elif isinstance(branch, QCheckBox):
            branch.setChecked(branch.origvalue)
        elif isinstance(branch, QSpinBox):
            branch.setValue(branch.origvalue)
        elif isinstance(branch, QLineEdit):
            branch.setText(str(branch.origvalue))

    def save(self):
        common = self.changed.pop('common', {})
        localchanges = {**self.changed.pop('gui', {}), **self.changed.pop(bs.net.server_id, {}), **common}
        remotechanges = dict(self.changed)
        self.changed.clear()

        # TODO: send possible changes to other servers

        # Save local settings
        success, msg = bs.settings.save(changes=localchanges)
        # stack.echo(msg)

    def btnclicked(self):
        self.resetbtn.setEnabled(False)
        self.savebtn.setEnabled(False)
        self.changedlabel.setText('')
        if self.sender() is self.savebtn:
            self.save()
        elif self.sender() is self.resetbtn:
            self.reset(self.scrollarea.layout)

    def hideEvent(self, event):
        pass

    def showEvent(self, event):
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

def clear_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
