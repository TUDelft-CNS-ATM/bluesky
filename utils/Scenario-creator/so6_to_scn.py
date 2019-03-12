import numpy as np
import collections
import textwrap
from datetime import datetime
# from bluesky.tools.aero import qdrdist, ft, tas2cas

class readFile:

    def __init__(self, datalocation, fileNumber):
        self.acid = []
        self.type = []
        self.lat = np.array([])
        self.lon = np.array([])
        self.hdg = np.array([])
        self.cas = np.array([])
        self.alt = np.array([])
        self.alt_end = np.array([])
        self.vs = np.array([])
        self.file = datalocation
        self.date = 120727
        self.time = []
        self.fileLocation = 'tempData/s06datfile_' + fileNumber + '.dat'
        self.writeFile = open(self.fileLocation, 'w')

    def remove_row_dates(self, arr, col, val):
        return arr[arr[col] == val]

    def openFile(self):
        names = ('origin', 'destination', 'actype', 't_begin', 't_end', 'fl_begin', 'fl_end', 'status', 'callsign', 'date_begin', 'date_end', 'lat_begin', 'lon_begin', 'lat_end', 'lon_end', 'flightid', 'sequence', 'length')
        ndtype = {'names': names,
         'formats': ('|S10',
                     '|S10',
                     '|S10',
                     '|S10',
                     '|S10',
                     int,
                     int,
                     int,
                     '|S10',
                     int,
                     int,
                     np.float16,
                     np.float16,
                     np.float16,
                     np.float16,
                     int,
                     int,
                     np.float16)}
        usecols = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
        air_traffic = np.loadtxt(self.file, usecols=usecols, dtype=ndtype)
        air_traffic.sort(order=['date_begin', 't_begin'])
        aircraft = np.unique(air_traffic['callsign'])
        aircraft = aircraft.tolist()
        return (air_traffic, aircraft)

    def setVariables(self, air_traffic, aircraft):
        for i in range(0, len(air_traffic)):
            self.date = air_traffic['date_begin'][i]
            self.date = str(self.date)
            self.acid = air_traffic['callsign'][i]
            self.type = air_traffic['actype'][i]
            self.lat = air_traffic['lat_begin'][i] / 60
            self.lon = air_traffic['lon_begin'][i] / 60
            lat2 = air_traffic['lat_end'][i] / 60
            lon2 = air_traffic['lon_end'][i] / 60
            self.alt = air_traffic['fl_begin'][i] * 100
            self.alt_end = air_traffic['fl_end'][i] * 100
            # self.hdg, distance = qdrdist(self.lat, self.lon, lat2, lon2)
            time_format = '%H:%M:%S'
            self.time = air_traffic['t_begin'][i]
            self.time = ':'.join(textwrap.wrap(self.time, 2))
            t_end = air_traffic['t_end'][i]
            t_end = ':'.join(textwrap.wrap(t_end, 2))
            t_segment = datetime.strptime(t_end, time_format) - datetime.strptime(self.time, time_format)
            t_segment = abs(t_segment.total_seconds()) + 0.001
            self.vs = 60.0 / float(t_segment) * (air_traffic['fl_end'][i] * 100 - self.alt)
            self.cas = 3600.0 / float(t_segment) * distance
            k = 0
            for k in range(17):
                year = '20' + str(self.date[0:2])
                year = int(year)
                month = int(self.date[2:4])
                day = int(self.date[4:6])
                seconds = int(self.time[6:8])
                minutes = int(self.time[3:5])
                hours = int(self.time[0:2])
                logTime = datetime(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds)
                currentTimeString = logTime.strftime('%Y-%m-%d %H:%M:%S')
                if k == 0:
                    startLine = str('=====================================================================') + '\n'
                    self.writeFile.write(startLine)
                    startLine = 'This is measurement; ' + str(i) + '\n'
                    self.writeFile.write(startLine)
                if k == 1:
                    line = 'Aircraft Type; ' + self.type + '\n'
                    self.writeFile.write(line)
                if k == 2:
                    line = 'Aircraft Registration' + '; ' + self.acid + '\n'
                    self.writeFile.write(line)
                if k == 3:
                    line = 'Date?; ' + currentTimeString + '\n'
                    self.writeFile.write(line)
                if k == 4:
                    line = 'Aircraft Callsign; ' + self.acid + '\n'
                    self.writeFile.write(line)
                if k == 5:
                    long = self.lon
                    line = 'Longitude; ' + str(int) + '\n'
                    self.writeFile.write(line)
                if k == 6:
                    lat = self.lat
                    line = 'Latitude; ' + str(lat) + '\n'
                    self.writeFile.write(line)
                if k == 7:
                    alt = self.alt
                    line = 'Altitude; ' + str(alt) + '\n'
                    self.writeFile.write(line)
                if k == 8:
                    line = 'PosTime; ' + currentTimeString + '\n'
                    self.writeFile.write(line)
                if k == 9:
                    line = 'VelTime; ' + currentTimeString + '\n'
                    self.writeFile.write(line)
                if k == 10:
                    cas = self.cas
                    line = 'Groundspeed; ' + str(cas) + '\n'
                    self.writeFile.write(line)
                if k == 11:
                    qdr = self.hdg
                    line = 'Track?; ' + str(qdr) + '\n'
                    self.writeFile.write(line)
                if k == 12:
                    line = 'Runway?; NaN\n'
                    self.writeFile.write(line)
                if k == 13:
                    line = 'FlightType; NaN\n'
                    self.writeFile.write(line)
                if k == 14:
                    line = 'VR?; NaN\n'
                    self.writeFile.write(line)
                if k == 15:
                    line = 'GNSS?; NaN\n'
                    self.writeFile.write(line)
                if k == 16:
                    line = 'NUCR?; NaN\n'
                    self.writeFile.write(line)
                if k == 16:
                    line = 'Typecode?; NaN\n'
                    self.writeFile.write(line)

        self.writeFile.close()

    def time2s(self, time):
        time = [ time[i:i + 2] for i in range(0, len(time), 2) ]
        t = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
        print(t)
        return t
