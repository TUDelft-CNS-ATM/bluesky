try:
    # Try Qt4 first
    from PyQt4.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
except ImportError:
    # Else PyQt5 imports
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
import numpy as np
import time

# Local imports
from ...ui.qtgl import ACDataEvent, PanZoomEvent


class ScreenIO(QObject):
    # =========================================================================
    # Settings
    # =========================================================================
    # Update rate of simulation info messages [Hz]
    siminfo_rate = 2

    # Update rate of aircraft update messages [Hz]
    acupdate_rate = 5

    # =========================================================================
    # Signals
    # =========================================================================
    signal_siminfo = pyqtSignal(float, float, float, int, int)  # signal contents: sys freq, simdt, simt, n_ac, mode
    signal_update_aircraft = pyqtSignal(ACDataEvent)
    signal_show_filedialog = pyqtSignal()
    signal_display_text = pyqtSignal(str)
    signal_panzoom = pyqtSignal(PanZoomEvent)

    # =========================================================================
    # Slots
    # =========================================================================
    @pyqtSlot(str)
    def callback_userinput(self, command):
        self.sim.stack.stack(str(command))

    @pyqtSlot()
    def send_siminfo(self):
        t = time.time()
        dt = t - self.prevtime
        self.signal_siminfo.emit((self.sim.samplecount - self.prevcount) / dt, self.sim.simdt, self.sim.simt, self.sim.traf.ntraf, self.sim.mode)
        self.prevtime = t
        self.prevcount = self.sim.samplecount

    @pyqtSlot()
    def send_aircraft_data(self):
        data = ACDataEvent()
        data.ids = list(self.sim.traf.id)
        data.lat = np.array(self.sim.traf.lat, dtype=np.float32, copy=True)
        data.lon = np.array(self.sim.traf.lon, dtype=np.float32, copy=True)
        data.alt = np.array(self.sim.traf.alt, copy=True)
        data.tas = np.array(self.sim.traf.tas, copy=True)
        data.trk = np.array(self.sim.traf.trk, dtype=np.float32, copy=True)

        self.signal_update_aircraft.emit(data)

    # =========================================================================
    # Functions
    # =========================================================================
    def __init__(self, sim):
        super(ScreenIO, self).__init__()

        # Keep reference to parent simulation object for access to simulation data
        self.sim = sim

        # Timing bookkeeping counters
        self.prevtime = 0.0
        self.prevcount = 0

        # Output event timers
        self.siminfo_timer = QTimer()
        self.siminfo_timer.timeout.connect(self.send_siminfo)
        self.siminfo_timer.start(1000/self.siminfo_rate)

        self.acupdate_timer = QTimer()
        self.acupdate_timer.timeout.connect(self.send_aircraft_data)
        self.acupdate_timer.start(1000/self.acupdate_rate)

    def moveToThread(self, target_thread):
        self.siminfo_timer.moveToThread(target_thread)
        self.acupdate_timer.moveToThread(target_thread)
        super(ScreenIO, self).moveToThread(target_thread)

    def echo(self, text):
        self.signal_display_text.emit(text)

    def zoom(self, zoomfac):
        self.signal_panzoom.emit(PanZoomEvent(PanZoomEvent.Zoom, zoomfac))

    def pan(self, pan, absolute=False):
        if absolute:
            self.signal_panzoom.emit(PanZoomEvent(PanZoomEvent.PanAbsolute, pan))
        else:
            self.signal_panzoom.emit(PanZoomEvent(PanZoomEvent.Pan, pan))

    def show_file_dialog(self):
        self.signal_show_filedialog.emit()
        return ''
