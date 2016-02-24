try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPen, QColor, QFont
    from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItemGroup
except ImportError:
    from PyQt4.QtGui import QPen, QColor, QFont, QGraphicsView, QGraphicsScene, QGraphicsItemGroup


class AMANDisplay(QGraphicsView):
    def __init__(self):
        super(AMANDisplay, self).__init__()
        self.setGeometry(0, 0, 500, 600)
        self.setStyleSheet('background-color:#233370')
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene = QGraphicsScene()

        # Timeline boundaries
        pen = QPen(QColor('white'))
        self.scene.addLine(220, 0, 220, 600, pen)
        self.scene.addLine(280, 0, 280, 600, pen)
        self.scene.addLine(0, 30, 500, 30, pen)
        self.scene.addLine(220, 570, 280, 570, pen)

        # Timeline scale
        self.timeline = QGraphicsItemGroup()
        for y in range(0, 600, 15):
            self.timeline.addToGroup(self.scene.addLine(220, y, 230, y, pen))
            self.timeline.addToGroup(self.scene.addLine(270, y, 280, y, pen))

        self.scene.addItem(self.timeline)
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
