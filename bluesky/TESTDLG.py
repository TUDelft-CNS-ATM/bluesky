import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

def window():
   app = QApplication(sys.argv)
   w = QWidget()
   btn = QPushButton(w)
   btn.setText("Hello World!")
   btn.move(100,50)
   btn.clicked.connect(showdialog)
   w.setWindowTitle("PyQt Dialog demo")
   w.show()
   sys.exit(app.exec_())

def showdialog():
   dlg = QDialog()
   b1 = QPushButton("ok",dlg)
   b1.move(50,50)
   dlg.setWindowTitle("Dialog")
   dlg.setWindowModality(Qt.ApplicationModal)
   dlg.exec_()

if __name__ == '__main__':
   window()