try:
    from PyQt5.QtCore import QUrl, QFileInfo
    from PyQt5.QtWidgets import QVBoxLayout, QWidget, QPushButton
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
except:
    from PyQt4.QtCore import QUrl, QFileInfo
    from PyQt4.QtGui import QVBoxLayout, QWidget, QPushButton
    from PyQt4.QtWebKit import QWebView as QWebEngineView


class DocView(QWebEngineView):
    def __init__(self, parent=None):
        super(DocView, self).__init__(parent)

        class DocPage(QWebEnginePage):
            def __init__(self, parent=None):
                super(DocPage, self).__init__(parent)

            def acceptNavigationRequest(self, url, navtype, ismainframe):
                if navtype == self.NavigationTypeLinkClicked:
                    if url.url()[:6].lower() == 'stack:':
                        print 'test'
                        DocWindow.app.stack(url.url()[6:].lower())
                        return False

                return True
        self.page = DocPage()
        self.setPage(self.page)


class DocWindow(QWidget):
    app = None

    def __init__(self, app):
        super(DocWindow, self).__init__()
        DocWindow.app = app
        self.vlayout  = QVBoxLayout()
        self.view     = DocView()
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
