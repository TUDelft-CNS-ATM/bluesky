from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class CalendarDialog(QDialog):

    def __init__(self, parent):
        super().__init__(parent)
        self.cal = QCalendarWidget(self)

        self.resize(300, 300)
        self.cal.resize(300, 300)

class Example(QMainWindow):

    def __init__(self):
        super().__init__()
        self.resize(400, 200)

        toolBar = QToolBar(self)

        calendarAction = QAction(QIcon('test.png'), 'Calendar', self)
        calendarAction.triggered.connect(self.openCalendar)
        toolBar.addAction(calendarAction)

    def openCalendar(self):
        self.calendarWidget = CalendarDialog(self)
        self.calendarWidget.show()


app = QApplication([])

ex = Example()
ex.show()

app.exec_()