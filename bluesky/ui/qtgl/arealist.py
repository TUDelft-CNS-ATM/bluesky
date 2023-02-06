from math import cos, radians
try:
    from PyQt5.QtWidgets import QListView, QStyledItemDelegate, QStyle, QApplication as app
    from PyQt5.QtGui import QColor, QPalette
    from PyQt5.QtCore import QSize, Qt, QRect, QAbstractListModel, QModelIndex, Qt, QVariant
except ImportError:
    from PyQt6.QtWidgets import QListView, QStyledItemDelegate, QStyle, QApplication as app
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtCore import QSize, Qt, QRect, QAbstractListModel, QModelIndex, Qt, QVariant

from bluesky.network import sharedstate, context as ctx
from bluesky.core import Base, remotestore as rs
from bluesky.ui import palette
import bluesky as bs


class AreaModel(QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polynames = list()
        self.polydata = list()

    def reset(self):
        ''' Clear all data in model. '''
        self.beginResetModel()
        # self.items.clear()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.polynames)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            # if role == Qt.ItemDataRole.DisplayRole:
            return QVariant((self.polynames[index.row()], self.polydata[index.row()]))
        return QVariant()

    def append(self, items):
        polydata = items.get('polys', dict())
        polydata = {k:v for k, v in polydata.items() if k not in self.polynames}
        if polydata:
            self.beginInsertRows(QModelIndex(), len(self.polynames), len(self.polynames) + len(polydata) - 1)
            self.polynames.extend(list(polydata.keys()))
            self.polydata.extend(list(polydata.values()))
            self.endInsertRows()

    def itemsRemoved(self, items):
        # remove items from the list
        pass


class AreaList(Base, QListView):
    model: AreaModel = rs.ActData(group='arealist')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackgroundRole(QPalette.ColorRole.NoRole)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: transparent')
        self.setItemDelegate(AreaItem())
        self.setMouseTracking(True)
        self.setSpacing(3)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setSelectionBehavior(QListView.SelectionBehavior.SelectRows)

    def setModel(self, model):
        super().setModel(model)
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

    @sharedstate.subscriber(topic='POLY')
    def on_poly_update(self, data):
        listmodel: AreaModel = rs.get(ctx.sender_id, 'arealist').model
        if (ctx.action == ctx.action.ActChange or
            ctx.action == ctx.action.Reset and ctx.sender_id == bs.net.act_id):
            # On reset a new model is created, and on actchange the model
            # needs to be switched
            self.setModel(listmodel)

        elif ctx.action in (ctx.action.Append, ctx.action.Update):
            listmodel.append(ctx.action_content)
            


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
        # else:
        #     return QStyledItemDelegate.sizeHint(self, option, index)



# try:
#     from PyQt5.QtWidgets import QVBoxLayout, QLabel, QSizePolicy, QWidget
# except ImportError:
#     from PyQt6.QtWidgets import QVBoxLayout, QLabel, QSizePolicy, QWidget

# from bluesky.network import sharedstate, context as ctx
# from bluesky.ui import palette


# class AreaList(QWidget):
#     instance = None
#     areas = dict()

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         AreaList.instance = self
#         self.layout = QVBoxLayout(self)
#         self.setLayout(self.layout)

#     @staticmethod
#     @sharedstate.subscriber(topic='POLY')
#     def on_poly_update(data):
#         if ctx.action == ctx.action.Reset or ctx.action == ctx.action.ActChange:# TODO hack
#             # Simulation reset: Clear all entries
#             while AreaList.areas:
#                 AreaList.instance.layout.removeWidget(AreaList.areas.popitem()[1])
#             return
#         if ctx.action == ctx.action.ActChange:
#             print('Active node change detected by AreaList')
#         changed = ctx.action_content.get('polys')

#         for name in changed:
#             pdata = data.polys.get(name)
#             area = AreaList.areas.get(name)
#             if area and not pdata:
#                 # Deleted area
#                 AreaList.instance.layout.removeWidget(area)

#             if area is None:
#                 area = QLabel(AreaList.instance)
#                 p = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
#                 area.setSizePolicy(p)
#                 area.setMinimumHeight(60)
#                 AreaList.instance.layout.addWidget(area)
#                 AreaList.areas[name] = area
#             color = '#' + ''.join([f'{c:02x}' for c in pdata.get('color', palette.polys)])
#             area.setStyleSheet(f'background-color: white; border: 1px solid "{color}"; border-left-width: 8px solid "{color}"; color: black')
#             area.setText(f"<b>ID:</b> {name}<br><b>Class:</b>{pdata['shape']}")
