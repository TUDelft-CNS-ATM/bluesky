from functools import partial
import numpy as np

from PyQt6.QtWidgets import QTableView, QStyledItemDelegate, QStyle, QApplication as app
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import QSize, Qt, QRect, QAbstractTableModel, QModelIndex, Qt, QVariant, QTimer

from bluesky.core import Base, remotestore as rs
from bluesky.network import sharedstate as ss
from bluesky.tools import aero


class TrafficModel(Base, QAbstractTableModel):
    id: list = rs.ActData(group='acdata')
    alt: np.ndarray = rs.ActData(0, group='acdata')
    trk: np.ndarray = rs.ActData(0, group='acdata')
    cas: np.ndarray = rs.ActData(0, group='acdata')

    def __init__(self) -> None:
        super().__init__()
        self.naircraft = 0

        timer = QTimer(self)
        timer.timeout.connect(self.notifyUpdate)
        timer.start(1000)

    def notifyUpdate(self):
        if self.naircraft:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount(), self.columnCount()))

    def rowCount(self, parent=QModelIndex()):
        return len(self.id)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def headerData(self, section, orientation, role):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return ('ALT', 'TRK', 'CAS')[section]
        else:
            # TODO: traffic resizing can give read errors if length reduces
            return self.id[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            # if role == Qt.ItemDataRole.DisplayRole:
            idx = index.row()
            col = index.column()
            if col == 0:
                return f'FL{self.alt[idx] / aero.ft / 100:1.0f}'
            elif col == 1:
                return f'{self.trk[idx]:1.0f}'
            else:
                return f'{self.cas[idx] / aero.kts:1.0f}'

    @ss.subscriber(topic='ACDATA', actonly=True)
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


class TrafficList(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.setModel(TrafficModel())
        self.setBackgroundRole(QPalette.ColorRole.NoRole)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: transparent')
        # self.set
    


    