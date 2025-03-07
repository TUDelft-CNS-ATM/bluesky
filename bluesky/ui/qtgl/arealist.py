from math import cos, radians
from PyQt6.QtWidgets import QListView, QStyledItemDelegate, QStyle, QApplication as app
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import QSize, Qt, QRect, QAbstractListModel, QModelIndex, Qt, QVariant

from bluesky.network import context as ctx
from bluesky.network.subscriber import subscriber
from bluesky.network.sharedstate import ActData
from bluesky.core import Base
from bluesky.ui import palette
import bluesky as bs


class AreaModel(QAbstractListModel, Base):
    polys: ActData[dict] = ActData(group='poly')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polynames = list()

    def rowCount(self, parent=QModelIndex()):
        return len(self.polynames)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            # if role == Qt.ItemDataRole.DisplayRole:
            name = self.polynames[index.row()]
            data = self.polys.get(name)
            if data:
                return QVariant((name, self.polys[name]))
        return QVariant()

    def itemsRemoved(self, items):
        # remove items from the list
        pass

    @subscriber(topic='POLY')
    def on_poly_update(self, data):
        if ctx.action in (ctx.action.ActChange, ctx.action.Reset):
            # Notify the gui that the entire list needs to be updated
            self.beginResetModel()
            self.polynames = list(self.polys)
            self.endResetModel()

        elif ctx.action in (ctx.action.Append, ctx.action.Update):
            newpolys = ctx.action_content.get('polys', dict())
            newpolys = [name for name in newpolys if name not in self.polynames]
            if newpolys:
                self.beginInsertRows(QModelIndex(), len(self.polynames), len(self.polynames) + len(newpolys) - 1)
                self.polynames.extend(newpolys)
                self.endInsertRows()


class AreaList(QListView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setModel(AreaModel())
        self.setBackgroundRole(QPalette.ColorRole.NoRole)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: transparent')
        self.setItemDelegate(AreaItem())
        self.setMouseTracking(True)
        self.setSpacing(3)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setSelectionBehavior(QListView.SelectionBehavior.SelectRows)
        self.selectionModel().selectionChanged.connect(
                self.on_change_selection
        )

    def on_change_selection(self) -> None:
        indexes = self.selectedIndexes()
        data = indexes[0].data()[1]
        coords = data['coordinates']
        lat = coords[::2]
        lon = coords[1::2]
        latrange = (min(lat), max(lat))
        lonrange = (min(lon), max(lon))
        pan = (0.5 * sum(latrange), 0.5 * sum(lonrange))
        flat_earth = cos(radians(pan[0]))
        latrange = latrange[1] - latrange[0]
        lonrange = (lonrange[1] - lonrange[0]) * flat_earth
        zoom = 1 / (max(latrange, lonrange))
        bs.stack.stack(f'PAN {pan[0]} {pan[1]};ZOOM {zoom}')


class AreaItem(QStyledItemDelegate):
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
        name, data = index.data()

        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(Qt.GlobalColor.red)
        else:
            painter.setPen(txtcolor)
   
        if option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(option.rect, bghover)
        else:
            painter.fillRect(option.rect, bgcolor)
        r = QRect(option.rect)
        r.setWidth(10)

        color = QColor(*data.get('color', palette.polys), 255)
        painter.fillRect(r, color)
        # f"<b>ID:</b> {name}<br><b>Class:</b>{pdata['shape']}"
        items = [
            ('ID :', name),
            ('CLASS: ', data['shape'])
        ]
        r = QRect(option.rect)
        r.setLeft(r.left() + 15)
        for header, content in items:
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(r, Qt.AlignmentFlag.AlignLeft, header)
            bb = painter.boundingRect(option.rect, Qt.AlignmentFlag.AlignLeft, header)
            font.setBold(False)
            painter.setFont(font)
            r.setLeft(r.left() + bb.width())
            painter.drawText(r, Qt.AlignmentFlag.AlignLeft, content)
            r.setLeft(option.rect.left() + 15)
            r.setTop(r.top() + bb.height() + 4)
        # else:
        #     QStyledItemDelegate.paint(self, painter, option, index)

    def sizeHint(self, option, index):
        """ Returns the size needed to display the item in a QSize object. """
        # index.data()
        return QSize(200, 45)

