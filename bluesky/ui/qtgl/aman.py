try: 
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPen, QBrush, QColor, QFont
    from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItemGroup
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPen, QBrush, QColor, QFont
    from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItemGroup


class AMANDisplay(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 500, 600)
        self.setStyleSheet('background-color:#233370')
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene = QGraphicsScene(0, 0, 500, 600)

        # Timeline boundaries
        pen = QPen(QColor('white'))
        brush = QBrush(QColor('#233370'))
        self.scene.addLine(220, 0, 220, 600, pen)
        self.scene.addLine(280, 0, 280, 600, pen)
        self.scene.addLine(0, 30, 500, 30, pen)

        timelinebox = self.scene.addRect(220, 30, 60, 540, pen, brush)
        timelinebox.setFlag(timelinebox.ItemClipsChildrenToShape, True)

        # Timeline scale
        self.timeline  = QGraphicsItemGroup()
        self.timeline.setParentItem(timelinebox)
        self.timeticks = []
        for i in range(40):
            y = 15 * i
            w = 6
            if i % 5 == 0:
                w = 10
                self.timeticks.append(self.scene.addText('%02d' % (40 - i), QFont('Courier', 10)))
                self.timeticks[-1].setPos(240, y - 10)
                self.timeticks[-1].setDefaultTextColor(QColor('white'))
                self.timeline.addToGroup(self.timeticks[-1])
            self.timeline.addToGroup(self.scene.addLine(220, y, 220 + w, y, pen))
            self.timeline.addToGroup(self.scene.addLine(280 - w, y, 280, y, pen))

        self.lrwy = self.scene.addText('18R', QFont('Arial', 20, 50))
        self.lrwy.setPos(1, 1)
        self.lrwy.setDefaultTextColor(QColor('white'))
        # Finalize
        self.setScene(self.scene)
        self.show()

    def update(self, simt, data):
        # data has: ids, iafs, eats, etas, delays, rwys, spdratios

        # First delete old labels
        for key, in self.aircraft.keys:
            if key not in data.ids:
                # del label
                del self.aircraft[key]

        for i in len(data.ids):
            if data.ids[i] not in self.aircraft:
                # self.aircraft[data.ids[i]] = QLabel()
                pass

            # Generate the new label text and position
            newtext = '<font color=Red>bla</font>'  # veranderen
            lbl = self.aircraft[data.ids[i]]
            lbl.setText(newtext)
            lbl.move(posx, posy)  # move in pixels
