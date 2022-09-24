""" Documentation window for the QTGL version of BlueSky."""
try:
    from PyQt5.QtCore import QUrl, QFileInfo
    from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel
    try:
        try:
            # Within PyQt5 there are different locations for QWebView and QWebPage,
            # depending on release version.
            from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView, QWebEnginePage as QWebPage
        except ImportError:
            from PyQt5.QtWebKitWidgets import QWebView, QWebPage

        class DocView(QWebView):
            def __init__(self, parent=None):
                super().__init__(parent)
                class DocPage(QWebPage):
                    def acceptNavigationRequest(self, url, navtype, ismainframe):
                        if navtype == self.NavigationTypeLinkClicked:
                            if url.url()[:6].lower() == 'stack:':
                                DocWindow.app.stack(url.url()[6:].lower())
                                return False
                        return True
                self.page = DocPage()
                self.setPage(self.page)
    except ImportError:
        DocView = None
        
except ImportError:
    from PyQt6.QtCore import QUrl, QFileInfo
    from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel
    try:
        try:
            # Within PyQt6 there are different locations for QWebView and QWebPage,
            # depending on release version.
            from PyQt6.QtWebEngineWidgets import QWebEngineView as QWebView
            from PyQt6.QtWebEngineCore import QWebEnginePage as QWebPage
        except ImportError:
            from PyQt6.QtWebEngineWidgets import QWebView
            from PyQt6.QtWebEngineCore import QWebPage

        class DocView(QWebView):
            def __init__(self, parent=None):
                super().__init__(parent)
                class DocPage(QWebPage):
                    def acceptNavigationRequest(self, url, navtype, ismainframe):
                        if navtype == self.NavigationType.NavigationTypeLinkClicked:
                            if url.url()[:6].lower() == 'stack:':
                                DocWindow.app.stack(url.url()[6:].lower())
                                return False
                        return True
                self.page = DocPage()
                self.setPage(self.page)
    except ImportError:
        DocView = None

import bluesky as bs


class DocWindow(QWidget):
    app = None

    def __init__(self, app):
        super().__init__()
        DocWindow.app = app
        self.vlayout  = QVBoxLayout()
        self.backbtn = QPushButton('Back')
        self.closebtn = QPushButton('Close')
        if DocView is not None:
            self.view = DocView()
            self.backbtn.clicked.connect(self.view.back)
        else:
            self.view = QLabel('BlueSky was not able to initialize it\'s\n' +
                ' QtWebEngine-based documentation viewer.\n' +
                'There may be something wrong with your Qt installation.\n' +
                'If you haven\'t yet, try installing PyQtWebEngine:\n\n' +
                '    pip install PyQtWebEngine\n\n' +
                'or, if you don\'t use pip, install it with your preferred\n' +
                'python package manager.')
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
        self.setLayout(self.vlayout)
        self.setWindowTitle('BlueSky documentation')

    def show_cmd_doc(self, cmd):
        if not cmd:
            cmd = 'Command-Reference'
        if not isinstance(self.view, QLabel):
            fname = bs.resource(f'html/{cmd.lower()}.html').as_posix()
            self.view.load(QUrl.fromLocalFile(QFileInfo(fname).absoluteFilePath()))
