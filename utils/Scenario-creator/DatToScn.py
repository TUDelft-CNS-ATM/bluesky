import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pylab import *
import matplotlib.pyplot as plt
# import aero
from datetime import datetime, timedelta
from operator import itemgetter
from collections import OrderedDict

class DatToScn:

    def __init__(self, datalocation):
        self.datalocation = datalocation
        print(self.datalocation)
        self.f = open(self.datalocation, 'r')
        self.lines = self.f.readlines()
        self.f.close()
        self.ignoreTime = timedelta(minutes=15)
        self.sortedListValue = []
        self.sortedListLabel = []
        for line in self.lines:
            text = line.lower()
            if text[0] != '=' and not text.strip() == '':
                words = text.split(';')
                label = words[0].strip()
                value = words[1].strip()
                if label == 'longitude' or label == 'latitude' or label == 'altitude' or label == 'groundspeed' or label == 'track?' or label == 'vr?' or label == 'gnss?' or label == 'nucr?' or label == 'typecode?':
                    value = float(value)
                self.sortedListValue.append(value)
                self.sortedListLabel.append(label)

        self.acTypeArray = []
        i = 1
        while i < len(self.sortedListValue):
            self.acTypeArray.append(self.sortedListValue[i])
            i = i + 18

        self.acRegArray = []
        i = 2
        while i < len(self.sortedListValue):
            self.acRegArray.append(self.sortedListValue[i])
            i = i + 18

        self.dateArray = []
        i = 3
        while i < len(self.sortedListValue):
            self.dateArray.append(self.sortedListValue[i])
            i = i + 18

        self.acCallSignArray = []
        i = 4
        while i < len(self.sortedListValue):
            self.acCallSignArray.append(self.sortedListValue[i])
            i = i + 18

        self.longitudeArray = []
        i = 5
        while i < len(self.sortedListValue):
            self.longitudeArray.append(self.sortedListValue[i])
            i = i + 18

        self.longitudeArray = np.array(self.longitudeArray)
        self.latitudeArray = []
        i = 6
        while i < len(self.sortedListValue):
            self.latitudeArray.append(self.sortedListValue[i])
            i = i + 18

        self.latitude = np.array(self.latitudeArray)
        self.altitudeArray = []
        i = 7
        while i < len(self.sortedListValue):
            self.altitudeArray.append(self.sortedListValue[i])
            i = i + 18

        self.altitudeArray = np.array(self.altitudeArray)
        print(self.altitudeArray)
        self.posTimeArray = []
        i = 8
        while i < len(self.sortedListValue):
            self.posTimeArray.append(self.sortedListValue[i])
            i = i + 18

        self.posTimeArray = np.array(self.posTimeArray)
        self.velTimeArray = []
        i = 9
        while i < len(self.sortedListValue):
            self.velTimeArray.append(self.sortedListValue[i])
            i = i + 18

        self.velTimeArray = np.array(self.velTimeArray)
        self.groundSpeedArray = []
        i = 10
        while i < len(self.sortedListValue):
            self.groundSpeedArray.append(self.sortedListValue[i])
            i = i + 18

        self.groundSpeedArray = np.array(self.groundSpeedArray)
        self.trackArray = []
        i = 11
        while i < len(self.sortedListValue):
            self.trackArray.append(self.sortedListValue[i])
            i = i + 18

        self.runwayArray = []
        i = 12
        while i < len(self.sortedListValue):
            self.runwayArray.append(self.sortedListValue[i])
            i = i + 18

        self.flightTypeArray = []
        i = 13
        while i < len(self.sortedListValue):
            self.flightTypeArray.append(self.sortedListValue[i])
            i = i + 18

        self.vrArray = []
        i = 14
        while i < len(self.sortedListValue):
            self.vrArray.append(self.sortedListValue[i])
            i = i + 18

        self.gnssArray = []
        i = 15
        while i < len(self.sortedListValue):
            self.gnssArray.append(self.sortedListValue[i])
            i = i + 18

        self.nucrArray = []
        i = 16
        while i < len(self.sortedListValue):
            self.nucrArray.append(self.sortedListValue[i])
            i = i + 18

        self.typecodeArray = []
        i = 17
        while i < len(self.sortedListValue):
            self.typecodeArray.append(self.sortedListValue[i])
            i = i + 18

        k = 0
        self.tempTime = list(range(len(self.posTimeArray)))
        while k < len(self.posTimeArray):
            self.tempTime[k] = datetime.strptime(self.posTimeArray[k], '%Y-%m-%d %H:%M:%S')
            k = k + 1

        self.unsortedTimeDict = dict(list(zip(list(range(len(self.posTimeArray))), self.tempTime)))
        self.sortedDict = sorted(list(self.unsortedTimeDict.items()), key=itemgetter(1))
        self.sortedIndexes = list(range(len(self.sortedDict)))
        i = 0
        while i < len(self.sortedIndexes):
            temp = self.sortedDict[i]
            temp = temp[0]
            self.sortedIndexes[i] = temp
            i = i + 1

        self.acTypeArray = self.orderListToTime(self.acTypeArray, self.sortedIndexes)
        self.acRegArray = self.orderListToTime(self.acRegArray, self.sortedIndexes)
        self.dateArray = self.orderListToTime(self.dateArray, self.sortedIndexes)
        self.acCallSignArray = self.orderListToTime(self.acCallSignArray, self.sortedIndexes)
        self.longitudeArray = self.orderListToTime(self.longitudeArray, self.sortedIndexes)
        self.latitudeArray = self.orderListToTime(self.latitudeArray, self.sortedIndexes)
        self.altitudeArray = self.orderListToTime(self.altitudeArray, self.sortedIndexes)
        self.posTimeArray = self.orderListToTime(self.posTimeArray, self.sortedIndexes)
        self.velTimeArray = self.orderListToTime(self.velTimeArray, self.sortedIndexes)
        self.groundSpeedArray = self.orderListToTime(self.groundSpeedArray, self.sortedIndexes)
        self.trackArray = self.orderListToTime(self.trackArray, self.sortedIndexes)
        self.runwayArray = self.orderListToTime(self.runwayArray, self.sortedIndexes)
        self.flightTypeArray = self.orderListToTime(self.flightTypeArray, self.sortedIndexes)
        self.vrArray = self.orderListToTime(self.vrArray, self.sortedIndexes)
        self.gnssArray = self.orderListToTime(self.gnssArray, self.sortedIndexes)
        self.nucrArray = self.orderListToTime(self.nucrArray, self.sortedIndexes)
        self.typecodeArray = self.orderListToTime(self.typecodeArray, self.sortedIndexes)
        print('Creating scenario file...')
        scnFileName = 'tempData/ScenarioFile'
        self.writeFile = open(scnFileName + '.scn', 'w')
        self.createdAircraft = list()
        self.acIgnored = False
        i = 0
        while i < len(self.acTypeArray):
            if len(self.createdAircraft) == 0:
                self.acAlreadyCreated = False
            elif self.acCallSignArray[i] in self.createdAircraft:
                self.acAlreadyCreated = True
            else:
                self.acAlreadyCreated = False
            if self.acAlreadyCreated == False:
                date = self.posTimeArray[i]
                time = str(date[11:19])
                date = date[0:19]
                date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                acid = str(self.acCallSignArray[i])
                type = str(self.acTypeArray[i])
                lat = str(self.latitudeArray[i])
                lon = str(self.longitudeArray[i])
                qdr = '0'
                alt = str(self.altitudeArray[i])
                spd = '0'
                createLine = time + '.00>CRE ' + acid + ',' + type + ',' + lat + ',' + lon + ',' + qdr + ',' + alt + ',' + spd + '\n'
                self.writeFile.write(createLine)
                self.createdAircraft.append(acid)
                self.acIgnored = self.AcIgnoreCheck(i)
                if self.acIgnored == True:
                    deleteLine = time + '.00>DEL ' + acid + '\n'
                    self.writeFile.write(deleteLine)
                    try:
                        self.createdAircraft.remove(acid)
                    except:
                        print('Error: A/C could not be deleted from createdAircraft()' + acid)

            elif self.latitudeArray[i] == 0:
                self.acIgnored = self.AcIgnoreCheck(i)
                date = self.posTimeArray[i]
                time = str(date[11:19])
                if self.acIgnored == False:
                    acid = str(self.acCallSignArray[i])
                    spd = str(self.groundSpeedArray[i])
                    speedLine = time + '.00>' + acid + ' ' + 'SPD' + ' ' + spd + '\n'
                    self.writeFile.write(speedLine)
                    qdr = str(self.trackArray[i])
                    qdrLine = time + '.00>' + acid + ' ' + 'HDG' + ' ' + qdr + '\n'
                    self.writeFile.write(qdrLine)
                if self.acIgnored == True:
                    acid = str(self.acCallSignArray[i])
                    deleteLine = time + '.00>DEL ' + acid + '\n'
                    self.writeFile.write(deleteLine)
                    try:
                        self.createdAircraft.remove(acid)
                    except:
                        print('Error: A/C removal' + acid)

            elif self.groundSpeedArray[i] == 0:
                self.acIgnored = self.AcIgnoreCheck(i)
                date = self.posTimeArray[i]
                time = str(date[11:19])
                if self.acIgnored == False:
                    acid = str(self.acCallSignArray[i])
                    lat = str(self.latitudeArray[i])
                    lon = str(self.longitudeArray[i])
                    alt = str(self.altitudeArray[i])
                    moveLine = time + '.00>MOVE ' + acid + ',' + lat + ',' + lon + ',' + alt + '\n'
                    self.writeFile.write(moveLine)
                if self.acIgnored == True:
                    acid = str(self.acCallSignArray[i])
                    deleteLine = time + '.00>DEL ' + acid + '\n'
                    self.writeFile.write(deleteLine)
                    try:
                        self.createdAircraft.remove(acid)
                    except:
                        print('Error: A/C removal' + acid)

            elif self.groundSpeedArray[i] != 0 and self.latitudeArray != 0:
                self.acIgnored = self.AcIgnoreCheck(i)
                date = self.posTimeArray[i]
                time = str(date[11:19])
                if self.acIgnored == False:
                    acid = str(self.acCallSignArray[i])
                    lat = str(self.latitudeArray[i])
                    lon = str(self.longitudeArray[i])
                    alt = str(self.altitudeArray[i])
                    moveLine = time + '.00>MOVE ' + acid + ',' + lat + ',' + lon + ',' + alt + '\n'
                    self.writeFile.write(moveLine)
                    spd = str(self.groundSpeedArray[i])
                    speedLine = time + '.00>' + acid + ' ' + 'SPD' + ' ' + spd + '\n'
                    self.writeFile.write(speedLine)
                    qdr = str(self.trackArray[i])
                    qdrLine = time + '.00>' + acid + ' ' + 'HDG' + ' ' + qdr + '\n'
                    self.writeFile.write(qdrLine)
                if self.acIgnored == True:
                    acid = str(self.acCallSignArray[i])
                    deleteLine = time + '.00>DEL ' + acid + '\n'
                    self.writeFile.write(deleteLine)
                    try:
                        self.createdAircraft.remove(acid)
                    except:
                        print('Error: A/C removal' + acid)

            i = i + 1

        print('Scenario file created')

    def orderListToTime(self, listToBeOrdered, listWithSortedIndexes):
        orderedList = list(range(len(listWithSortedIndexes)))
        i = 0
        while i < len(listWithSortedIndexes):
            index = listWithSortedIndexes[i]
            orderedList[i] = listToBeOrdered[index]
            i = i + 1

        return orderedList

    def AcIgnoreCheck(self, aircraftIndex):
        k = aircraftIndex + 1
        ignoredBoolean = 'test'
        while k < len(self.acCallSignArray):
            if self.acCallSignArray[k] == self.acCallSignArray[aircraftIndex]:
                time1 = self.posTimeArray[aircraftIndex]
                time1 = time1[0:19]
                time1 = datetime.strptime(time1, '%Y-%m-%d %H:%M:%S')
                time2 = self.posTimeArray[k]
                time2 = time2[0:19]
                time2 = datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')
                ignoreTimeDifference = time1 + self.ignoreTime
                if ignoreTimeDifference < time2:
                    ignoredBoolean = True
                else:
                    ignoredBoolean = False
                k = len(self.acCallSignArray)
            else:
                k = k + 1
                ignoredBoolean = True

        return ignoredBoolean
