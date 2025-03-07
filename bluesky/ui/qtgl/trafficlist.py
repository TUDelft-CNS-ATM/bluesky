import numpy as np

from PyQt6.QtWidgets import QListView, QStyledItemDelegate, QApplication as app
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import QSize, Qt, QRect, QAbstractListModel, QModelIndex, Qt, QVariant, QTimer

from bluesky.core import Base
from bluesky.network.subscriber import subscriber
from bluesky.network.sharedstate import ActData
from bluesky.tools import aero


class TrafficModel(QAbstractListModel, Base):
    id: ActData[list] = ActData(group='acdata')
    alt: ActData[np.ndarray] = ActData(0, group='acdata')
    trk: ActData[np.ndarray] = ActData(0, group='acdata')
    cas: ActData[np.ndarray] = ActData(0, group='acdata')

    def __init__(self) -> None:
        super().__init__()
        self.naircraft = 0

        timer = QTimer(self)
        timer.timeout.connect(self.notifyUpdate)
        timer.start(1000)

    def notifyUpdate(self):
        if self.naircraft:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount(), 0))

    def rowCount(self, parent=QModelIndex()):
        return len(self.id)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            # if role == Qt.ItemDataRole.DisplayRole:
            idx = index.row()
            # Qt is multi-threaded, so data model size can change during view update
            if idx >= len(self.id):
                return QVariant()
            return QVariant((
                self.id[idx],
                f'FL{self.alt[idx] / aero.ft / 100:1.0f}',
                f'{self.trk[idx]:1.0f}',
                f'{self.cas[idx] / aero.kts:1.0f}'
            ))

    @subscriber(topic='ACDATA', actonly=True)
    def on_data_update(self, data):
        if len(data.id) == 0:
            self.beginResetModel()
            self.endResetModel()
        elif len(data.id) > self.naircraft:
            self.beginInsertRows(QModelIndex(), self.naircraft, len(data.id) - 1)
            self.endInsertRows()
        elif len(data.id) < self.naircraft:
            diff = self.naircraft - len(data.id)
            self.beginRemoveRows(QModelIndex(), self.naircraft - diff - 1, self.naircraft - 1)
            self.endRemoveRows()
        self.naircraft = len(data.id)


class TrafficList(QListView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setModel(TrafficModel())
        self.setBackgroundRole(QPalette.ColorRole.NoRole)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: transparent')
        self.setItemDelegate(TrafficItem())
    

class TrafficItem(QStyledItemDelegate):
    """ AreaItem is used to render area items in the area list """
    def paint(self, painter, option, index):
        # Determine dark mode
        p = app.instance().style().standardPalette()
        isdark = (p.color(p.ColorRole.Window).value() < p.color(p.ColorRole.WindowText).value())

        if isdark:
            bgcolor = p.color(p.ColorRole.Window).lighter(120)
            bghover = p.color(p.ColorRole.Window).lighter(140)
            txtcolor = Qt.GlobalColor.white
        else:
            bgcolor = QColor(250, 250, 250, 255)
            bghover = Qt.GlobalColor.lightGray
            txtcolor = Qt.GlobalColor.black

        # Get polygon name and data
        acid, alt, trk, cas = index.data()

        r = QRect(option.rect)
        r.setLeft(r.left() + 15)

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(r, Qt.AlignmentFlag.AlignLeft, acid)
        bb = painter.boundingRect(option.rect, Qt.AlignmentFlag.AlignLeft, acid)
        font.setBold(False)
        painter.setFont(font)
        r.setLeft(r.left() + bb.width())
        painter.drawText(r, Qt.AlignmentFlag.AlignLeft, ' '.join((alt, trk, cas)))

    def sizeHint(self, option, index):
        """ Returns the size needed to display the item in a QSize object. """
        # index.data()
        return QSize(200, 25)
    