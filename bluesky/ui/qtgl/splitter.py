from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QSplitterHandle, QVBoxLayout, QToolButton


class Handle(QSplitterHandle):
    ''' Custom splitter handle with collapse button. '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vl = QVBoxLayout(self)
        self.vl.setContentsMargins(0, 0, 0, 0)
        self.b = QToolButton(self)
        self.b.setArrowType(Qt.ArrowType.LeftArrow)
        self.b.clicked.connect(self.parent().collapse)
        self.vl.addWidget(self.b)
        self.setLayout(self.vl)


class Splitter(QSplitter):
    ''' Custom splitter with collapse button.
    
        A Splitter divides resizeable panes within a window. 
    '''
    def __init__(self, *args, defwidth=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.prevstate = [10000, defwidth]
        self.splitterMoved.connect(self.on_splitter_move)

    def createHandle(self):
        return Handle(self.orientation(), self)

    def collapse(self):
        ''' Callback for the collapse button on the Splitter handle. '''
        sizes = self.sizes()
        if sizes[-1] == 0:
            sizes = self.prevstate
            self.handle(1).b.setArrowType(Qt.ArrowType.RightArrow)

        else:
            self.prevstate = sizes[:]
            sizes[-1] = 0
            self.handle(1).b.setArrowType(Qt.ArrowType.LeftArrow)
        self.setSizes(sizes)

    def on_splitter_move(self, pos, index):
        ''' Keep track of Splitter position to update handle appearance. '''
        handle = self.handle(1)
        if self.width() - handle.width() - pos == 0:
            handle.b.setArrowType(Qt.ArrowType.LeftArrow)
        else:
            handle.b.setArrowType(Qt.ArrowType.RightArrow)
