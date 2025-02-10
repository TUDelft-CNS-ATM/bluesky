from PyQt6.QtWidgets import QListView
from PyQt6.QtGui import QPalette


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