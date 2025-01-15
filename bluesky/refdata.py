''' Reference values for simulation data. '''
from bluesky.core.base import Base
from bluesky import stack
from bluesky.tools import areafilter


class RefData(Base):
    def __init__(self) -> None:
        super().__init__()
        self.lat: float = 0.0
        self.lon: float = 0.0
        self.alt: float = 0.0
        self.acidx: int = -1
        self.hdg: float = 0.0
        self.cas: float = 0.0
        self.area: areafilter.Shape = areafilter.Box('refarea', (-1.0, -1.0, 1.0, 1.0))

    def reset(self):
        ''' Reset reference data. '''
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.acidx = -1
        self.hdg = 0.0
        self.cas = 0.0
        self.area = areafilter.Box('refarea', (-1.0, -1.0, 1.0, 1.0))

    @stack.command
    def near(self, lat: 'lat', lon: 'lon', cmdstr: 'string'):
        '''Set reference lat/lon before executing command string. '''
        self.lat = lat
        self.lon = lon
        stack.process(cmdstr)

    @stack.commandgroup
    def inside(self, lat0: 'lat', lon0: 'lon', lat1: 'lat', lon1: 'lon', cmdstr: 'string'):
        self.area = areafilter.Box('refarea', (lat0, lon0, lat1, lon1))
        stack.process(cmdstr)

    @inside.altcommand
    def insidearea(self, areaname: 'txt', cmdstr: 'string'):
        self.area = areafilter.getArea(areaname)
        stack.process(cmdstr)

    @stack.command(aliases=('with',))
    def withac(self, idx: 'acid', cmdstr: 'string'):
        self.acidx = idx
        stack.process(cmdstr)
