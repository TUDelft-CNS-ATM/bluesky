""" Documentation window for the QTGL version of BlueSky."""
from PyQt5.QtCore import QUrl, QFileInfo
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QPushButton
try:
    # Within PyQt5 there are different locations for QWebView and QWebPage,
    # depending on release version.
    from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView, QWebEnginePage as QWebPage
except ImportError:
    from PyQt5.QtWebKitWidgets import QWebView, QWebPage

class DocView(QWebView):
    def __init__(self, parent=None):
        super(DocView, self).__init__(parent)

        class DocPage(QWebPage):
            def __init__(self, parent=None):
                super(DocPage, self).__init__(parent)

            def acceptNavigationRequest(self, url, navtype, ismainframe):
                if navtype == self.NavigationTypeLinkClicked:
                    if url.url()[:6].lower() == 'stack:':
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
        self.backbtn  = QPushButton('Back')
        self.closebtn = QPushButton('Close')
        self.vlayout.setContentsMargins(1, 1, 1, 1)
        self.vlayout.setSpacing(1)
        self.vlayout.addWidget(self.view)
        hlayout = QHBoxLayout()
        buttonbox = QWidget()
        buttonbox.setLayout(hlayout)
        self.vlayout.addWidget(buttonbox)
        hlayout.addWidget(self.closebtn)
        hlayout.addWidget(self.backbtn)
        self.closebtn.clicked.connect(self.hide)
        self.backbtn.clicked.connect(self.view.back)
        self.setLayout(self.vlayout)
        self.setWindowTitle('BlueSky documentation')

    def show_cmd_doc(self, cmd):
        if not cmd:
            cmd = 'Command-Reference'
        self.view.load(QUrl.fromLocalFile(QFileInfo('data/html/' + cmd.lower() + '.html').absoluteFilePath()))
