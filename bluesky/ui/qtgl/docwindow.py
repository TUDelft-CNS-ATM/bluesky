try:
    from PyQt5.QtCore import QUrl, QFileInfo
    from PyQt5.QtWidgets import QVBoxLayout, QWidget, QPushButton
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except:
    from PyQt4.QtCore import QUrl, QFileInfo
    from PyQt4.QtGui import QVBoxLayout, QWidget, QPushButton
    from PyQt4.QtWebKit import QWebView as QWebEngineView


class DocWindow(QWidget):
    def __init__(self):
        super(DocWindow, self).__init__()
        self.vlayout  = QVBoxLayout()
        self.view     = QWebEngineView()
        self.closebtn = QPushButton('Close')
        self.vlayout.setContentsMargins(1, 1, 1, 1)
        self.vlayout.setSpacing(1)
        self.vlayout.addWidget(self.view)
        self.vlayout.addWidget(self.closebtn)
        self.closebtn.clicked.connect(self.hide)
        self.setLayout(self.vlayout)
        self.setWindowTitle('BlueSky documentation')

    def show_cmd_doc(self, cmd):
        if not cmd:
            cmd = 'Command-Reference'
        self.view.load(QUrl.fromLocalFile(QFileInfo('data/html/' + cmd.lower() + '.html').absoluteFilePath()))
