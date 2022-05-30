"""
This converts PH-LAB log data to the .dat format, which in its turn can be converted to a .scn file

Created by  : Yoeri Torel (TU Delft)
Date        : March 2014

Modification:
By          :
Date        :

"""
import scipy.io
import math
from datetime import datetime
from pathlib import Path

class CitationToDatFile:

    def __init__(self, datalocation):
        self.datalocation = datalocation
        directory = Path('tempData/citationData')
        if not directory.is_dir():
            directory.mkdir()
        self.matlabFile = self.datalocation
        self.tempStr = self.matlabFile[len(self.matlabFile) - 14:]
        self.tempStr = self.tempStr[:len(self.tempStr) - 4]
        self.storeFile = 'tempData/citationData/FlightTestData_' + self.tempStr + '.dat'
        self.mat = scipy.io.loadmat(self.matlabFile, squeeze_me=True)
        self.tab = self.mat['ptr'][()]
        self.indexMatFile = []
        self.writeFile = open(self.storeFile, 'w')
        for j in range(len(self.tab[0])):
            for i in range(17):
                year = int(self.tempStr[4:8])
                month = int(self.tempStr[2:4])
                day = int(self.tempStr[0:2])
                hours = int(self.tab[68][j])
                minutes = int(self.tab[69][j])
                seconds = int(self.tab[70][j])
                logTime = datetime(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds)
                currentTimeString = logTime.strftime('%Y-%m-%d %H:%M:%S')
                if i == 0:
                    startLine = str('=====================================================================') + '\n'
                    self.writeFile.write(startLine)
                    startLine = 'This is measurement; ' + str(j + 1) + '\n'
                    self.writeFile.write(startLine)
                if i == 1:
                    line = 'Aircraft Type; ' + 'CitationII' + '\n'
                    self.writeFile.write(line)
                if i == 2:
                    line = 'Aircraft Registration' + '; ' + 'PH-LAB' + '\n'
                    self.writeFile.write(line)
                if i == 3:
                    line = 'Date?; ' + currentTimeString + '\n'
                    self.writeFile.write(line)
                if i == 4:
                    line = 'Aircraft Callsign; ' + 'PH-LAB' + '\n'
                    self.writeFile.write(line)
                if i == 5:
                    long = self.tab[42][j]
                    long = float(float(int * 180) / 1048576)
                    line = 'Longitude; ' + str(int) + '\n'
                    self.writeFile.write(line)
                if i == 6:
                    lat = self.tab[41][j]
                    lat = float(float(lat * 180) / 1048576)
                    line = 'Latitude; ' + str(lat) + '\n'
                    self.writeFile.write(line)
                if i == 7:
                    Palt = self.tab[22][j]
                    Balt = self.tab[23][j]
                    line = 'Altitude; ' + str(Palt) + '\n'
                    self.writeFile.write(line)
                if i == 8:
                    line = 'PosTime; ' + currentTimeString + '\n'
                    self.writeFile.write(line)
                if i == 9:
                    line = 'VelTime; ' + currentTimeString + '\n'
                    self.writeFile.write(line)
                if i == 10:
                    ias = self.tab[25][j]
                    line = 'Groundspeed; ' + str(ias) + '\n'
                    self.writeFile.write(line)
                if i == 11:
                    qdr = self.tab[43][j]
                    qdr = qdr / math.pi * 180
                    line = 'Track?; ' + str(qdr) + '\n'
                    self.writeFile.write(line)
                if i == 12:
                    line = 'Runway?; NaN\n'
                    self.writeFile.write(line)
                if i == 13:
                    line = 'FlightType; NaN\n'
                    self.writeFile.write(line)
                if i == 14:
                    line = 'VR?; NaN\n'
                    self.writeFile.write(line)
                if i == 15:
                    line = 'GNSS?; NaN\n'
                    self.writeFile.write(line)
                if i == 16:
                    line = 'NUCR?; NaN\n'
                    self.writeFile.write(line)
                if i == 16:
                    line = 'Typecode?; NaN\n'
                    self.writeFile.write(line)

        self.writeFile.close()

    def __repr__(self):
        return self.storeFile

