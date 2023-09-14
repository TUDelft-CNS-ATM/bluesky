try:
    from PyQt5.QtWidgets import QListView, QStyledItemDelegate, QStyle, QApplication as app
    from PyQt5.QtGui import QColor, QPalette
    from PyQt5.QtCore import QSize, Qt, QRect, QAbstractListModel, QModelIndex, Qt, QVariant
except ImportError:
    from PyQt6.QtWidgets import QListView, QStyledItemDelegate, QStyle, QApplication as app
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtCore import QSize, Qt, QRect, QAbstractListModel, QModelIndex, Qt, QVariant



class StackList(QListView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('StackList')
        self.setBackgroundRole(QPalette.ColorRole.NoRole)
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: transparent')
    
    def keyPressEvent(self, event) -> None:
        print('stacklist')
        return super().keyPressEvent(event)