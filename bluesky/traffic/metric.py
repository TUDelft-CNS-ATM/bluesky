from time import time, gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt
from math import degrees
import collections
from collections import defaultdict
import itertools as IT

import bluesky as bs
from bluesky.tools import geo
from bluesky.tools.misc import tim2txt
from bluesky.tools.aero import *
from bluesky import settings

# Register settings defaults
settings.set_variable_defaults(log_path='output')

"""
    This module seems to work as follows:
    It looks like this used to be a submodule of the traffic module
    Now it is a full module under the sim package

    Apparently the creation of this module also called into life the concept of a research area.
    A research area is something which is not specific to the metrics module and could be used by different modules.
    However, the area command in the stack module saves data in the metric instance.
    If the area function and metric module are to be seperatly used, then they should be untangled.

    Classes:
    Metric, metric_Area, metric_CoCa, metric_HB
    Each has a constructor
    Requirements for instance creation:
        Metric: -
        metric_Area: -
        metric_CoCa: regions
        metric_HB: area

    Called from:
        metric_Area: Metric
        metric_CoCa: Metric
        metric_HB: Metric

    Passed as argument to:
        metric_Area: metric_CoCa, metric_HB (only the .cellArea instance)
        metric_CoCa: -
        metric_HB: -

    Structure:
        metric object created with Metric class constructor
        metric.metric object is of the form (metric_CoCa, metric_HB)
            So the metric instance has two sub instances,
            one of type metric_CoCa and one of type metric_HB
"""
class metric_Area():
    def __init__(self):
        self.lat = 55.5
        self.lon = 1.7
        self.fll = 8500
        self.flu = 41500
        self.bearingS = 180
        self.bearingE = 90
        self.deltaFL = 3000
        self.distance = 20
        self.ncells = 18
        self.nlevels = 12

        self.regions = np.array([0,0,0])

    def addbox(self,lat,lon):

        lat_0 = lat
        lat_00 = lat
        lon_0 = lon
        londiviser = 1
        for i in range(1,self.ncells+1):
            for j in range(1,self.ncells+1):
                for k in range(self.fll,self.flu+self.deltaFL,self.deltaFL):
                    box = np.array([lat,lon,k])
                    self.regions = np.vstack([self.regions,box])

                if i == 1:

                    lat,lon = geo.qdrpos(lat,lon,self.bearingE,self.distance)
                    lat = degrees(lat)
                    lon = degrees(lon)
                    londiviser = (lon - lon_0) / self.ncells
                else:
                    lat,lon = geo.qdrpos(lat,lon,self.bearingE,self.distance)
                    lat = degrees(lat)
                    lon = lon_0 + londiviser * j

            lat_0 = lat_00
            lat,lon = geo.qdrpos(lat_0,lon_0,self.bearingS,self.distance*i)
            lat = degrees(lat)
            lon = degrees(lon)
            lat_0 = lat

        return

    def cellArea(self):
        point1 = [self.regions[0,0],self.regions[0,1]]
        point2 = [self.regions[self.ncells*self.nlevels-1,0],self.regions[self.ncells*self.nlevels-1,1]]
        point3 = [self.regions[(self.ncells-1)*self.ncells*self.nlevels,0],self.regions[(self.ncells-1)*self.ncells*self.nlevels,1]]
        point4 = [self.regions[self.ncells*self.ncells*self.nlevels-1,0],self.regions[self.ncells*self.ncells*self.nlevels-1,1]]
        self.cellarea = np.array([point4,point2,point1,point3])
        #print self.cellarea
        return self.cellarea

    def makeRegions(self):
        lat = self.lat
        lon = self.lon

        self.addbox(lat,lon)
        self.regions = np.delete(self.regions, (0), axis=0)

        return self.regions


    def area_of_polygon(self,x, y):
        area = 0.0
        for i in range(-1, len(x) - 1):
            area += x[i] * (y[i + 1] - y[i - 1])
        return area / 2.0

    def centroid_of_polygon(self,points):
        area = self.area_of_polygon(*list(zip(*points)))

        result_x = 0
        result_y = 0
        N = len(points)

        points = IT.cycle(points)

        x1, y1 = next(points)
        for i in range(N):
            x0, y0 = x1, y1
            x1, y1 = next(points)
            cross = (x0 * y1) - (x1 * y0)
            result_x += (x0 + x1) * cross
            result_y += (y0 + y1) * cross
        result_x /= (area * 6.0)
        result_y /= (area * 6.0)
        return (result_x, result_y)


    def FIR_circle(self, fir_number):
        fir_lat = []
        fir_lon = []
        fir = []

        fir_lat.append(bs.navdb.fir[fir_number][1])
        fir_lon.append(bs.navdb.fir[fir_number][2])
        fir.append((fir_lat[-1],fir_lon[-1]))

        fir = fir[0]
        fir = list(zip(fir[0],fir[1]))
        fir_centroid = self.centroid_of_polygon(fir)

        return fir_centroid

class metric_CoCa():

    def __init__(self,regions):


        self.region = regions
        self.oldaircraft = np.zeros((1000,1), dtype = [('callsign','|S10'),('cellnumber',int), ('time',int),('totaltime',int)])
        self.newaircraft = np.zeros((1000,1), dtype = [('callsign','|S10'),('cellnumber',int), ('time',int),('totaltime',int)])
        # self.cells = np.zeros((self.region.nlevels*self.region.ncells*self.region.ncells,1), dtype = [('cellnumber',int),('interactions',int),('ntraf',int)])
        # for i in range(0,len(self.cells)):
        #    self.cells['cellnumber'][i] = i + 1
        #        plt.close()

        self.numberofcells = self.region.ncells*self.region.ncells*self.region.nlevels

        names = []
        for i in range(0,self.numberofcells):
            names.append("cell"+str(i))

        formats = []
        for i in range(0,self.numberofcells):
            formats.append("|S10")

        ndtype = {'names':names, 'formats':formats}
        self.cells = np.zeros((500,6), dtype = ndtype)
        self.resettime = 5 #seconds
        self.deltaresettime = self.resettime
        self.iteration = 0

        formats = []
        for i in range(0,self.numberofcells):
            formats.append(float)

        ndtype = {'names':names, 'formats':formats}
        oneday = 86400 # second in one day
        numberofrows = oneday / self.resettime
        numberofrows = 3
        self.precocametric = np.zeros((numberofrows,5), dtype = ndtype)
        self.cocametric = np.zeros((numberofrows,6), dtype = ndtype)
        plt.ion()
        self.ntraf = 0

        # plt.colorbar()
        # self.plotntraf,= plt.plot([], [])
        # self.plotbar, = plt.bar([],[])
        return

    def findCell(self,cells,lat,lon,fl):
        i = 0
        j = 0
        k = 0

        for i in range(0,len(cells),self.region.ncells*self.region.nlevels):
            if (cells[0,0]) <= lat < (cells[0,0]+0.6):
                break
            if (cells[i,0] < lat < cells[0,0] and cells[i,0] > cells[-1,0]) :
                break
            else:
                i = -10000

        if i > -1 :

            for j in range(0,self.region.ncells*self.region.nlevels,self.region.nlevels):

                if cells[i+j,1] > lon and lon < (cells[-1,1]+0.6) and lon > cells[0,1]:
                    j = j - self.region.nlevels
                    break
                if cells[i+j,1]+0.6 > lon and lon < (cells[-1,1]+0.6) and lon > cells[0,1]:
                    break
                else:
                    j = -10000

            if j > - 1:
                for k in range(0,self.region.nlevels,1):
                    if cells[i+j+k,2] > fl and fl < (cells[-1,2]+self.region.deltaFL) and fl > cells[0,2]:
                        k = k -1
                        break
                    else:
                        k = -10000

        if (i+j+k) < 0:
            i=-1
            j=0
            k=0

        return i+j+k


    # def update_line(self,ntraf,t):
    #     if t < 0.1:

    #        self.__init__()

    #     t = int(t)
    #     self.plotntraf.set_xdata(np.append(self.plotntraf.get_xdata(), t))
    #     self.plotntraf.set_ydata(np.append(self.plotntraf.get_ydata(), ntraf))
    #     plt.plot(t,ntraf,'b--o')
    #     ax = plt.gca()
    #     ax.relim()
    #     ax.autoscale_view()
    #     return

    # def update_bar(self,trafcell,t):
    #    if t < 0.1:
    #        self.__init__
    #    t = int(t)
    #    self.plotbar.set_xdata(np.append(self.plotbar.get_xdata(), t))
    #    self.plotbar.set_ydata(np.append(self.plotbar.get_ydata(), trafcell))
    #    plt.plot(t,ntraf,'b--o')
    #    ax = plt.gca()
    #    ax.relim()
    #    ax.autoscale_view()
    #    return

    # def plot_interactions(self):
    #      plotcells = np.sort(self.cells, axis = 0, order='interactions')[-3:]
    #      label = np.vstack(plotcells['cellnumber'])
    #      x = np.arange(len(label))
    #      y1 = np.vstack(plotcells['interactions'])
    #      colLabels=("Interactions","")
    #      nrows, ncols = len(x)+1, len(colLabels)
    #      hcell, wcell = 0.3, 0.5
    #      hpad, wpad = 0, 0.5
    #      fig1=plt.figure(num = 1, figsize=(ncols*wcell+wpad, nrows*hcell+hpad))
    #      ax = fig1.add_subplot(111)
    #      ax.axis('off')
    #      ax.table(cellText=y1,colLabels=colLabels, rowLabels = label ,loc='center')
    #      plt.bar(x,y)
    #      plt.xticks(x,str(label))
    #      plt.show()
    #     return

    # def cell_interactions(self,cellN):
    #    # Interactions
    #    itemscount = np.array(collections.Counter(cellN).items())
    #    if len(itemscount) > 0:
    #        for number in range(0,self.region.ncells*self.region.ncells*self.region.nlevels):
    #            j = np.where(itemscount[:,0] == number)
    #            if np.size(j) == 0:
    #                self.cells['interactions'][number] = 0
    #            else:
    #                self.cells['interactions'][number] = itemscount[j,1]*(itemscount[j,1]-1)
    #        self.plot_interactions()
    #    return


    # def celltime(self,time):
    #     for i in range(0,len(self.newaircraft)):
    #         j = np.where(self.oldaircraft['callsign'] == self.newaircraft['callsign'][i])[0]
    #         if np.size(j) == 1:
    #             if self.oldaircraft['cellnumber'][j] == self.newaircraft['cellnumber'][i]:
    #                 self.newaircraft['time'][i] = time - self.oldaircraft['totaltime'][i]
    #             else:
    #                 self.newaircraft['totaltime'][i] = time - self.oldaircraft['totaltime'][i]
    #                 self.cells['ntraf'][i] = self.cells['ntraf'][i] + 1
    #     self.oldaircraft = self.newaircraft
    #     return self.newaircraft


    def cellPlot(self):
        cell = [floor(x/12) for x in bs.traf.cell]
        count = collections.Counter(cell)
        count = np.array(list(count.items()))

        flcells = count
        if np.size(count)>0:
            flcells = count[:,0]

        z = np.array([0])
        for number in range(0,self.region.ncells*self.region.ncells):
            i = np.where(flcells == (number))

            if np.size(i) == 0:
                z = np.append(z,0)
            else:
                i = i[0]
                z = np.append(z,count[i,1])

        z = np.delete(z, (0), axis=0)
        zdata = np.reshape(z,(-1,self.region.ncells))
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(zdata, interpolation='nearest')
        plt.show()
        return

    def applyMetric(self):
        for i in range(0,self.numberofcells):
            name = 'cell'+str(i)

            l = self.iteration

            times = []
            headings = []
            speeds = []
            vspeeds = []
            actimes = []

            for j in range(0,len(self.cells[name])):
                if self.cells[name][j][1] != "":
                    times.append(float(self.cells[name][j][1]))
                    headings.append(float(self.cells[name][j][2]))
                    speeds.append(float(self.cells[name][j][3]))
                    vspeeds.append(float(self.cells[name][j][4]))
                    actimes.append(float(self.cells[name][j][1]))

            indices = np.argsort(times)
            times.sort()

            headings = [headings[z] for z in indices]
            speeds = [speeds[y] for y in indices]
            vspeeds = [vspeeds[x] for x in indices]
            actimes = [actimes[w] for w in indices]


            for w in range(0,len(vspeeds)):
                if vspeeds[w] <= 500 and vspeeds[w] >= (-500):
                    vspeeds[w] = 0
                elif vspeeds[w] > 500:
                    vspeeds[w] = 1
                elif vspeeds[w] < (-500):
                    vspeeds[w] = -1



            self.precocametric[name][l][0] = (sum(times)/self.deltaresettime)

            acinteractions = []
            spdinteractions = []
            hdginteractions = []
            vspdinteractions = []

            if len(times) > 1:
                for k in range(0,len(times)):
                    aircraft = len(times)

                    time = times[0]/self.deltaresettime
                    actime = actimes[0]/self.deltaresettime
                    acinteractions.append(aircraft*(aircraft-1)*(actime**aircraft))


                    counter = 0
                    for t in range(0,1):
                        for u in range(t+1,len(speeds)):
                            if abs(speeds[t]-speeds[u]) > 35:
                                counter = counter + 1
                    spdinteractions.append(2*counter*(time**(counter+1)))


                    counter = 0
                    for t in range(0,1):
                        for u in range(t+1,len(headings)):
                            if abs(headings[t]-headings[u]) > 20:
                                counter = counter + 1
                    hdginteractions.append(2*counter*(time**(counter+1)))


                    counter = 0
                    for t in range(0,1):
                        for u in range(t+1,len(vspeeds)):
                            if vspeeds[t] != vspeeds[u]:
                                counter = counter + 1
                    vspdinteractions.append(2*counter*(time**(counter+1)))

                    for x in range(1,len(actimes)):
                        actimes[x] = actimes[x] - actimes[0]

                    del actimes[0]
                    del times[0]
                    del vspeeds[0]
                    del speeds[0]
                    del headings[0]

                self.precocametric[name][l][1] = sum(acinteractions)
                self.precocametric[name][l][2] = sum(spdinteractions)
                self.precocametric[name][l][3] = sum(hdginteractions)
                self.precocametric[name][l][4] = sum(vspdinteractions)

                self.cocametric[name][l][1] = self.precocametric[name][l][1] / self.precocametric[name][l][0]
                self.cocametric[name][l][2] = self.precocametric[name][l][2] / self.precocametric[name][l][0]
                self.cocametric[name][l][3] = self.precocametric[name][l][3] / self.precocametric[name][l][0]
                self.cocametric[name][l][4] = self.precocametric[name][l][4] / self.precocametric[name][l][0]
                self.cocametric[name][l][0] = self.cocametric[name][l][1] * (self.cocametric[name][l][2] + self.cocametric[name][l][3] + self.cocametric[name][l][4])

        print("Iteration number: "+str(self.iteration+1))
        print("Reset time = "+str(self.resettime))
        return


    def reset(self):
        names = []
        for i in range(0,self.numberofcells):
            names.append("cell"+str(i))

        formats = []
        for i in range(0,self.numberofcells):
            formats.append("|S10")

        ndtype = {'names':names, 'formats':formats}
        self.cells = np.zeros((500,6), dtype = ndtype)

        return

    def AircraftCell(self,cells,time):
        if floor(time) >= self.resettime:
            bs.sim.hold()
            self.reset()
            self.resettime = self.resettime + self.deltaresettime
            self.iteration = self.iteration + 1
            # filedata = bs.resource(settings.log_path) / "coca_20120727-78am-1hour.npy"
            # self.cellPlot(traf)
            # np.save(filedata,self.cocametric)
            bs.sim.op()

        bs.traf.cell = []

        for i in range(bs.traf.ntraf):
            lat = bs.traf.lat[i]
            lon = bs.traf.lon[i]
            fl = bs.traf.alt[i]/ft
            cellN = self.findCell(cells,lat,lon,fl)

            if cellN > 0:
                bs.traf.cell = np.append(bs.traf.cell, cellN)
                name = 'cell'+str(cellN)

                index = np.where(bs.traf.id[i] == self.cells[name][:,[0]])[0]
                if len(index) != 1:

                    j = 0
                    for j in range(0,len(self.cells[name])):
                        if self.cells[name][j][0] == "":
                            break
                    self.cells[name][j][0] = bs.traf.id[i]
                    self.cells[name][j][1] = time
                    self.cells[name][j][2] = bs.traf.ahdg[i]
                    self.cells[name][j][3] = eas2tas(bs.traf.selspd[i],bs.traf.selalt[i])/kts
                    self.cells[name][j][4] = bs.traf.selvs[i]/fpm
                    self.cells[name][j][5] = time
                if len(index) == 1:
                    createtime = float(self.cells[name][index[0]][5])
                    self.cells[name][index[0]][1] = str(time - createtime)

                # self.newaircraft['callsign'][i] = bs.traf.id[i]
                # self.newaircraft['cellnumber'][i] =
        return


class metric_HB():

    def __init__(self,area):
        self.initiallat = area[3][0]
        self.initiallon = area[3][1]
        self.dist_range = 5.0 #nm

        self.alt_range = 1000.0 #ft
        self.t_cpa = 0
        self.dist_cpa = 0
        self.spd = np.array([])
        self.lat = np.array([])
        self.lon = np.array([])
        self.pos = np.array([])
        self.trk = 0

        self.alt_dif = 0
        self.alt = 0
        self.id = []

        self.complexity = defaultdict(lambda:defaultdict(int))
        self.rel_trk = np.array([])
        self.step = -1
        self.id_previous = []
        self.headings = []
        self.headings_previous = np.array([])
        self.doubleconflict = 0
        self.ntraf = 0
        self.compl_ac = 0
        self.time_lookahead = 1800 #seconds

        self.selected_area = ([area[0][0],area[0][1]],[area[1][0],area[1][1]],[area[2][0],area[2][1]],[area[3][0],area[3][1]])

        return

    def selectTraffic(self):

        traf_selected_lat = np.array([])
        traf_selected_lon = np.array([])
        traf_selected_alt = np.array([])
        traf_selected_tas = np.array([])
        traf_selected_trk = np.array([])
        traf_selected_ntraf = 0
        # RECTANGLE AREA
        # for i in range(0,bs.traf.ntraf):
        #     if nx.pnpoly(bs.traf.lat[i],bs.traf.lon[i],self.selected_area) == 1:
        #         traf_selected_lat = np.append(traf_selected_lat,bs.traf.lat[i])
        #         traf_selected_lon = np.append(traf_selected_lon,bs.traf.lon[i])
        #         traf_selected_alt = np.append(traf_selected_alt,bs.traf.alt[i])
        #         traf_selected_tas = np.append(traf_selected_tas,bs.traf.tas[i])
        #         traf_selected_trk = np.append( traf_selected_trk,bs.traf.trk[i])
        #         traf_selected_ntraf = traf_selected_ntraf + 1

        # CIRCLE AREA (FIR Circle)
        for i in range(0,bs.traf.ntraf):

            dist = latlondist(bs.sim.metric.fir_circle_point[0],\
                              bs.sim.metric.fir_circle_point[1],\
                              bs.traf.lat[i],bs.traf.lon[i])

            if  dist/nm < bs.sim.metric.fir_circle_radius:
                traf_selected_lat = np.append(traf_selected_lat,bs.traf.lat[i])
                traf_selected_lon = np.append(traf_selected_lon,bs.traf.lon[i])
                traf_selected_alt = np.append(traf_selected_alt,bs.traf.alt[i])
                traf_selected_tas = np.append(traf_selected_tas,bs.traf.tas[i])
                traf_selected_trk = np.append( traf_selected_trk,bs.traf.trk[i])
                traf_selected_ntraf = traf_selected_ntraf + 1


        return traf_selected_lat,traf_selected_lon,traf_selected_alt,traf_selected_tas,traf_selected_trk,traf_selected_ntraf


    def applymetric(self):
        time1 = time()
        bs.sim.hold()
        self.doubleconflict = 0
        # relative pos x and pos y
        self.step = self.step + 1
        self.pos = np.array([])
        self.lat = np.array([])
        self.lon = np.array([])

        self.id = []
        self.alt_dif = 0

        traf_selected_lat,traf_selected_lon,traf_selected_alt,traf_selected_tas,traf_selected_trk,traf_selected_ntraf = self.selectTraffic()


        [self.rel_trk, self.pos] = geo.qdrdist_matrix(self.initiallat,self.initiallon,np.mat(traf_selected_lat),np.mat(traf_selected_lon))
        # self.lat = np.append(self.lat,traf.lat)
        # self.lon = np.append(self.lon,traf.lon)
        self.id = bs.traf.id

        # Position x and y wrt to initial position
        self.pos = np.mat(self.pos)
        anglex = np.cos(np.radians(90-self.rel_trk))
        angley = np.sin(np.radians(90-self.rel_trk))

        self.posx = np.mat(np.array(self.pos) * np.array(anglex)) #nm
        self.posy = np.mat(np.array(self.pos) * np.array(angley)) #nm

        self.lat = traf_selected_lat
        self.lon = traf_selected_lon

        self.alt = np.mat(traf_selected_alt/ft)
        self.spd = traf_selected_tas/nm #nm/s
        self.trk = traf_selected_trk
        self.ntraf = traf_selected_ntraf

        self.alt_dif = self.alt-self.alt.T
        # Vectors CPA_dist and CPA_time
        self.apply_twoCircleMethod()
        time2 = time()
        print("Time to Complete Calculation: " + str(time2-time1))

        bs.sim.op()
        return


    def rel_matrixs(self):
        self.alt_dif = self.alt-self.alt.T
        # speeds
        hdgx = np.cos(np.radians(90-self.trk))
        hdgy = np.sin(np.radians(90-self.trk))

        spdu = np.mat(self.spd * hdgx.T).T #nm/s
        spdv = np.mat(self.spd * hdgy.T).T #nm/s

        # distances pos and spd
        distx = np.array(self.posx.T - self.posx) #nm
        disty = np.array(self.posy.T - self.posy) #nm
        distu = (np.array(spdu.T - spdu)) #nm/s
        distv = (np.array(spdv.T - spdv)) #nm/s

        # predicted time to CPA
        self.t_cpa = -(distu*distx+distv*disty)/      \
         (distu*distu+distv*distv+np.array(np.eye(distu[:,0].size)))

        # predicted distance to CPA
        relcpax = self.t_cpa*np.array(spdu.T)
        relcpay = self.t_cpa*np.array(spdv.T)
        cpax = self.posx.T + relcpax
        cpay = self.posy.T + relcpay
        distcpax = np.array(cpax-cpax.T)
        distcpay = np.array(cpay-cpay.T)
        self.dist_cpa = (distcpax**2+distcpay**2)**0.5

        return

    def apply_altfilter(self,S0):
        condition = abs(self.alt_dif)<self.alt_range
        # self.t_cpa = np.where(condition,self.t_cpa, np.nan)
        # self.dist_cpa = np.where(condition,self.dist_cpa, np.nan)
        S0 = np.where(condition,S0,np.nan)
        return S0

    def apply_distfilter(self,H0):
        condition = self.dist_cpa<self.dist_range*3
        self.dist_cpa = np.where(condition,self.dist_cpa,np.nan)
        self.t_cpa = np.where(condition,self.t_cpa,np.nan)
        H0 = np.where(condition,H0,np.nan)
        return H0

    def apply_timefilter(self):
        condition = self.t_cpa>0#(self.t_cpa<(self.time_range+20) * (self.t_cpa>0))
        self.t_cpa = np.where(condition,self.t_cpa,np.nan)
        self.dist_cpa = np.where(condition,self.dist_cpa,np.nan)
        return

    def apply_before_filter(self,S0,Va):
        Vb = Va.T
        Va_Vb = np.add(np.abs(Va),np.abs(Vb))
        condition1 = S0>0
        condition2 = np.divide(S0,Va_Vb)>self.time_lookahead #seconds
        condition = np.multiply(condition1,condition2)
        condition = np.invert(condition)
        S0 = np.where(condition,S0,np.nan)
        return S0

    def merge(self,times):
            if len(times) > 0:
                saved = list(times[0])

                for st, en in sorted([(t) for t in times]):
                    if st <= saved[1]:
                        saved[1] = max(saved[1], en)
                    else:
                        yield list(saved)
                        saved[0] = st
                        saved[1] = en
                yield list(saved)
            else:
                yield list(times)

    def apply_twoCircleMethod(self):
        Va = np.mat(self.spd)
        Ha = np.radians(self.trk)

        Vb = np.add(Va,0.0000001)
        VaVa = np.multiply(Va,Va)

        Hb = Ha

        [H0,S0] = geo.qdrdist_matrix(np.mat(self.lat),np.mat(self.lon),np.mat(self.lat),np.mat(self.lon))
        S0 = np.where(S0 > 0, S0, np.nan)

        S0 = self.apply_before_filter(S0,Va)
        S0 = self.apply_altfilter(S0)

        H0 = np.radians(H0.T)

        R_S0 = np.divide(self.dist_range,S0)
        arcsin = np.arcsin(R_S0)

        ha_new11,ha_new21,ha_new12,ha_new22,t1d1,t1d2,t2d1,t2d2 = self.calc_angles(Vb,Hb,VaVa,H0,arcsin,S0)

        R_S0 = None
        arcsin = None

        ha_1 = np.degrees(ha_new11)
        ha_3 = np.degrees(ha_new21)

        ha_2 = np.degrees(ha_new12)
        ha_4 = np.degrees(ha_new22)

        t1 = t1d1
        t2 = t1d2
        t3 = t2d1
        t4 = t2d2

        ha_new11 = None
        ha_new21 = None
        ha_new12 = None
        ha_new22 = None
        t1d1 = None
        t1d2 = None
        t2d1 = None
        t2d2 = None

        ha_1,ha_2,ha_3,ha_4,t1,t2,t3,t4 = self.conditions(ha_1,ha_2,ha_3,ha_4,t1,t2,t3,t4,Va,Vb,Ha,Hb)

        ## Condition where S0 < self.dist_range
        condition = np.multiply(S0<self.dist_range,S0>0)

        ac_angles = {}
        ac_score = {}
        for k in range(0,self.ntraf):
            ac_angles[str(k)] = []
            ac_score[str(k)] = 0
            for l in range(0,self.ntraf):
                if not np.isnan(ha_1[l,k]) and not np.isnan(ha_2[l,k]):
                    ac_angles[str(k)].append((ha_1[l,k],ha_2[l,k]))
                if not np.isnan(ha_3[l,k]) and not np.isnan(ha_4[l,k]):
                    ac_angles[str(k)].append((ha_3[l,k],ha_4[l,k]))

            ac_angles[str(k)] = sorted(ac_angles[str(k)])
            ac_angles[str(k)] = list(self.merge(ac_angles[str(k)]))
            if len(ac_angles[str(k)][0]) > 0:
                for z in range(0,len(ac_angles[str(k)])):
                    ac_angle180min = ((self.trk[k]+180-90)%360-180)
                    ac_angle180max = ((self.trk[k]+180+90)%360-180)
                    ac_angles_st180 = ((ac_angles[str(k)][z][0]+180)%360-180)
                    ac_angles_en180 = ((ac_angles[str(k)][z][-1]+180)%360-180)

                    ac_angle360min = (self.trk[k]+360-90)%360
                    ac_angle360max = (self.trk[k]+360+90)%360
                    ac_angles_st360 = (ac_angles_st180+360)%360
                    ac_angles_en360 = (ac_angles_en180+360)%360

                    if ac_angle180min<90 and ac_angle180min>-90:
                        if ac_angles_st180 < ac_angle180min:
                            ac_angles[str(k)][z][0] = ac_angle180min
                    else:
                        if ac_angles_st360 < ac_angle360min:
                            ac_angles[str(k)][z][0] = ac_angle180min

                    if ac_angle180max < 90 and ac_angle180max > -90:
                        if ac_angles_en180 > ac_angle180max:
                            ac_angles[str(k)][z][-1] = ac_angle180max
                    else:
                        if ac_angles_en360 > ac_angle360max:
                            ac_angles[str(k)][z][-1] = ac_angle180max

                    if (ac_angles_st180 < ac_angle180min and ac_angles_en180 < ac_angle180min) or (ac_angles_st360 < ac_angle360min and ac_angles_en360 < ac_angle360min):
                        ac_angles[str(k)][z] = [np.nan,np.nan]

                    # Complexity Score
                    if ac_angles[str(k)][z][-1]<90 and ac_angles[str(k)][z][-1]>-90:
                        ac_score[str(k)] = ac_score[str(k)] + (ac_angles[str(k)][z][-1]-ac_angles[str(k)][z][0])/180
                    else:
                        ac_score[str(k)] = ac_score[str(k)] + (ac_angles[str(k)][z][-1]+360-ac_angles[str(k)][z][0])/180

            if True in condition[k]:
                ac_score[str(k)] = 1

            if np.isnan(ac_score[str(k)]):
                ac_score[str(k)] = 0

        ac_totalscore =  sum(ac_score.values())

        self.complexity[self.step][0] = ac_totalscore #/ self.ntraf
        self.complexity[self.step][1] = ac_totalscore / max(1,self.ntraf)

        print("Complexity per Aircraft: " + str(self.complexity[self.step][1]))
        return


    def calc_angles(self,Vb,Hb,VaVa,H0,arcsin,S0):
        wx = np.multiply(Vb,np.sin(Hb))
        wy = np.multiply(Vb,np.cos(Hb))

        wxwx = np.multiply(wx,wx)
        wywy = np.multiply(wy,wy)
        wxwx_wywy = np.add(wxwx,wywy)

        a = np.subtract(wxwx_wywy.T,VaVa)

        H0_arcsin = np.subtract(H0,arcsin)
        xc1 = np.multiply(S0,np.sin(H0_arcsin))
        yc1 = np.multiply(S0,np.cos(H0_arcsin))

        xc1xc1 = np.multiply(xc1,xc1)
        yc1yc1 = np.multiply(yc1,yc1)

        a = np.where(a!=0,a,np.nan)
        a = np.add(a,0.00000000001)

        b1 = np.add(2*np.multiply(xc1,wx.T),2*np.multiply(yc1,wy.T))
        c1 = np.add(xc1xc1,yc1yc1)
        b1b1 = np.multiply(b1,b1)
        d1 = np.subtract(b1b1,4*np.multiply(a,c1))

        c1 = None
        b1b1 = None
        xc1xc1 = None
        yc1yc1 = None
        H0_arcsin = None

        a_2 = np.multiply(a,2)
        conditiond1 = d1<0
        conditiond1 = np.invert(conditiond1)
        d1 = np.where(conditiond1,d1,np.nan)
        t01d1 = np.divide(np.subtract(-b1,np.sqrt(d1)),a_2)
        t02d1 = np.divide(np.add(-b1,np.sqrt(d1)),a_2)

        t1d1 = np.minimum(t01d1,t02d1)
        t2d1 = np.maximum(t01d1,t02d1)

        xpt1d1 = np.add(xc1,np.multiply(wx.T,t1d1))
        xpt2d1 = np.add(xc1,np.multiply(wx.T,t2d1))

        ypt1d1 = np.add(yc1,np.multiply(wy.T,t1d1))
        ypt2d1 = np.add(yc1,np.multiply(wy.T,t2d1))

        xc1 = None
        yc1 = None

        H0_arcsin = np.add(H0,arcsin)
        xc2 = np.multiply(S0,np.sin(H0_arcsin))
        yc2 = np.multiply(S0,np.cos(H0_arcsin))

        xc2xc2 = np.multiply(xc2,xc2)
        yc2yc2 = np.multiply(yc2,yc2)

        wxT = wx.T
        wyT = wy.T
        xc2_wx = np.multiply(xc2,wxT)
        yc2_wy = np.multiply(yc2,wyT)
        xc2_wx2 = np.multiply(xc2_wx,2)
        yc2_wy2 = np.multiply(yc2_wy,2)
        b2 = np.add(xc2_wx2,yc2_wy2)
        c2 = np.add(xc2xc2,yc2yc2)
        b2b2 = np.multiply(b2,b2)
        d2 = np.subtract(b2b2,4*np.multiply(a,c2))

        c2 = None
        b2b2 = None
        xc2xc2 = None
        yc2yc2 = None
        H0_arcsin = None

        wxwx = None
        wywy = None
        wxwx_wywy = None

        conditiond2 = d2<0
        conditiond2 = np.invert(conditiond2)
        d2 = np.where(conditiond2,d2,np.nan)

        t01d2 = np.divide(np.subtract(-b2,np.sqrt(d2)),a_2)
        t02d2 = np.divide(np.add(-b2,np.sqrt(d2)),a_2)
        t1d2 = np.minimum(t01d2,t02d2)
        t2d2 = np.maximum(t01d2,t02d2)

        t01d1 = None
        t02d1 = None
        t01d2 = None
        t02d2 = None

        d1 = None
        d2 = None

        xpt1d2 = np.add(xc2,np.multiply(wx.T,t1d2))
        xpt2d2 = np.add(xc2,np.multiply(wx.T,t2d2))

        ypt1d2 = np.add(yc2,np.multiply(wy.T,t1d2))
        ypt2d2 = np.add(yc2,np.multiply(wy.T,t2d2))

        xc2 = None
        yc2 = None
        wx = None
        wy = None

        ha_new11 = np.arctan2(xpt1d1,ypt1d1)
        ha_new21 = np.arctan2(xpt2d1,ypt2d1)

        ha_new12 = np.arctan2(xpt1d2,ypt1d2)
        ha_new22 = np.arctan2(xpt2d2,ypt2d2)

        return ha_new11,ha_new21,ha_new12,ha_new22,t1d1,t1d2,t2d1,t2d2

    def conditions(self,ha_1,ha_2,ha_3,ha_4,t1,t2,t3,t4,Va,Vb,Ha,Hb):
        t1_nan = t1>0
        t2_nan = t2>0
        t3_nan = t3>0
        t4_nan = t4>0

        ha_1 = np.where(t1_nan,ha_1,np.nan)
        ha_2 = np.where(t2_nan,ha_2,np.nan)
        ha_3 = np.where(t3_nan,ha_3,np.nan)
        ha_4 = np.where(t4_nan,ha_4,np.nan)

        t1 = np.where(t1_nan,t1,np.nan)
        t2 = np.where(t2_nan,t2,np.nan)
        t3 = np.where(t3_nan,t3,np.nan)
        t4 = np.where(t4_nan,t4,np.nan)

        # condition: Va < Vb and all + t's
        Va_Vb = Va < Vb
        t1_t2 = np.multiply(t1_nan,t2_nan)
        t3_t4 = np.multiply(t3_nan,t4_nan)
        t_allplus = np.multiply(t1_t2,t3_t4)
        condition = np.multiply(Va_Vb,t_allplus)
        condition = np.invert(condition)
        ha_3new = np.where(condition,ha_3,ha_4)
        t3new = np.where(condition,t3,t4)
        ha_4new = np.where(condition,ha_4,ha_3)
        t4new = np.where(condition,t4,t3)

        # condition Va < Vb and t1,t3 negatif
        t1_neg = np.invert(t1_nan)
        t3_neg = np.invert(t3_nan)
        t1_t3_neg = np.multiply(t1_neg,t3_neg)
        condition = np.multiply(Va_Vb,t1_t3_neg)
        condition = np.invert(condition)
        ha_3 = np.where(condition,ha_3new,ha_4new)
        t3 = np.where(condition,t3new,t4new)
        ha_4 = np.where(condition,ha_4new,ha_2)
        t4 = np.where(condition,t4new,t2)

        # condition Va < Vb and t2,t4 negatif
        t2_neg = np.invert(t2_nan)
        t4_neg = np.invert(t4_nan)
        t2_t4_neg = np.multiply(t2_neg,t4_neg)
        condition = np.multiply(Va_Vb,t2_t4_neg)
        condition = np.invert(condition)
        ha_2 = np.where(condition,ha_2,ha_3)
        t2 = np.where(condition,t2,t3)

        # TBD More than 90-degree turns!
        Ha = np.degrees(Ha)
        Hb = np.degrees(Hb)

        # Lookahead time
        t1_lht = t1 > self.time_lookahead
        t2_lht = t2 > self.time_lookahead
        t1_t2_lht = np.multiply(t1_lht,t2_lht)
        t1_t2_lht = np.invert(t1_t2_lht)

        t3_lht = t3 > self.time_lookahead
        t4_lht = t4 > self.time_lookahead
        t3_t4_lht = np.multiply(t3_lht,t4_lht)
        t3_t4_lht = np.invert(t3_t4_lht)

        ha_1 = np.where(t1_t2_lht,ha_1,np.nan)
        ha_2 = np.where(t1_t2_lht,ha_2,np.nan)
        ha_3 = np.where(t3_t4_lht,ha_3,np.nan)
        ha_4 = np.where(t3_t4_lht,ha_4,np.nan)

        t1 = np.where(t1_t2_lht,t1,np.nan)
        t2 = np.where(t1_t2_lht,t2,np.nan)
        t3 = np.where(t3_t4_lht,t3,np.nan)
        t4 = np.where(t3_t4_lht,t4,np.nan)

        return ha_1,ha_2,ha_3,ha_4,t1,t2,t3,t4

    def saveData(self):
        acid = np.array(self.id).reshape(-1,).tolist()
        lat = np.array(self.lat).reshape(-1,).tolist()
        lon = np.array(self.lon).reshape(-1,).tolist()
        compl = np.array(self.compl_ac).reshape(-1,).tolist()
        alt =  np.array(self.alt).reshape(-1,).tolist()
        spd = np.array(self.spd).reshape(-1,).tolist()
        trk = np.array(self.trk).reshape(-1,).tolist()
        ntraf = np.repeat(np.array(self.ntraf),len(trk))
        ntraf = np.array(ntraf).reshape(-1,).tolist()

        data = zip(acid,lat,lon,alt,spd,trk,ntraf,compl)

        step = str(self.step).zfill(3)
        fname = bs.resource(settings.log_path) / "Metric-HB" / f"{step}-BlueSky.csv"
        f = csv.writer(open(fname, "wb"))
        for row in data:
            f.writerow(row)
        return

    def complexity_plot(self):
        if self.step == 0:
            self.plot_complexity,= plt.plot([], [])
            self.plot_complexity2, = plt.plot([],[])
            plt.ion()

        t = int(self.step)
        self.plot_complexity.set_xdata(np.append(self.plot_complexity.get_xdata(), t))
        self.plot_complexity.set_ydata(np.append(self.plot_complexity.get_ydata(), self.complexity[self.step][0]))
        self.plot_complexity2.set_xdata(np.append(self.plot_complexity2.get_xdata(), t))
        self.plot_complexity2.set_ydata(np.append(self.plot_complexity2.get_ydata(), self.complexity[self.step][1]))

        plt.subplot(2,1,1)
        plt.plot(t,self.complexity[self.step][0],'b--o--')
        plt.subplot(2,1,2)
        plt.plot(t,self.complexity[self.step][1],'g--o--')
        plt.draw()
        return


        #    def heading_range(self,i,ac1_conflict,ac2_conflict,lat0_ac1,lon0_ac1,lat0_ac2,lon0_ac2):
        #
        #        h_range = np.arange(int(self.trk[ac1_conflict[i]])-90,int(self.trk[ac1_conflict[i]])+90,1)
        #        h_range = np.append(h_range,self.trk[ac2_conflict[i]])
        #
        #
        #        hdgx = np.cos(np.radians(90-h_range))
        #        hdgy = np.sin(np.radians(90-h_range))
        #
        #        spd = np.array([])
        #        posy = np.array([])
        #        posx = np.array([])
        #        for l in range(0,len(h_range)-1):
        #            posy = np.append(posy,self.posy[ac1_conflict[i]])
        #            posx = np.append(posx,self.posx[ac1_conflict[i]])
        #            spd = np.append(spd,self.spd[ac1_conflict[i]])
        #
        #        spd = np.append(spd,self.spd[ac2_conflict[i]])
        #
        #        spdu = np.mat(spd * hdgx).T #nm/s
        #        spdv = np.mat(spd * hdgy).T #nm/s
        #        posy = np.append(posy,self.posy[ac2_conflict[i]])
        #        posx = np.append(posx,self.posx[ac2_conflict[i]])
        #        posx = np.mat(posx).T
        #        posy = np.mat(posy).T
        #
        #        ## distances pos and spd
        #        distx = np.array(posx - posx.T) #nm
        #        disty = np.array(posy - posy.T) #nm
        #        distu = (np.array(spdu - spdu.T)) #nm/s
        #        distv = (np.array(spdv - spdv.T)) #nm/s
        #
        #        ## predicted time to CPA
        #        t_heading = -(distu*distx+distv*disty)/      \
        #         (distu*distu+distv*distv+np.array(np.eye(distu[:,0].size)))
        #
        #        t_heading = self.apply_timefilter(t_heading)
        #
        #        relcpax = t_heading*np.array(spdu.T)
        #        relcpay = t_heading*np.array(spdv.T)
        #        cpax = posx.T + relcpax
        #        cpay = posy.T + relcpay
        #        distcpax = np.array(cpax-cpax.T)
        #        distcpay = np.array(cpay-cpay.T)
        #        distcpa = (distcpax**2+distcpay**2)**0.5
        #
        #        n_headings = np.where(distcpa>self.dist_range)
        #        headings = np.array([])
        #
        #        for z in range(0,len(n_headings[0])/2):
        #            headings = np.append(headings,h_range[n_headings[0][z]])
        #
        #
        #
        #
        #        #check if flightid is same, so multiple conflicts per id
        #        if self.id[ac1_conflict[i]] == self.id_previous:
        #            headings = np.intersect1d(self.headings_previous,headings)
        #            n_fault_headings = len(h_range)-len(headings)
        #            ratio = float(float(n_fault_headings)/float(len(h_range)))
        #            add_complexity = [self.id[ac1_conflict[i]],ratio,headings]
        #            self.complexity[self.step][i-self.doubleconflict-1] = add_complexity
        #            self.doubleconflict = self.doubleconflict + 1
        #        else:
        #            n_fault_headings = len(h_range)-len(headings)
        #            ratio = float(float(n_fault_headings)/float(len(h_range)))
        #            add_complexity = [self.id[ac1_conflict[i]],ratio,headings]
        #            self.complexity[self.step][i-self.doubleconflict] = add_complexity
        #
        #
        #        self.id_previous = self.id[ac1_conflict[i]]
        #        self.headings_previous = headings
        #
        #        return
        #
        #
        #
        #    def apply_precisetime(self,range_t,delta_t,heading_range):
        #
        #        ac_conflict = np.where(self.t_cpa>0)
        #        conflicts = np.size(ac_conflict)/2
        #
        #        if conflicts > 0:
        #            ac1_conflict = ac_conflict[0]
        #            ac2_conflict = ac_conflict[1]
        #
        #
        #            t_cpa_precise = np.array([])
        #            plat_ac1 = np.array([])
        #            plat_ac2 = np.array([])
        #            plon_ac1 = np.array([])
        #            plon_ac2 = np.array([])
        #
        #            for i in range(0,conflicts):
        #                lat0_ac1 = self.lat[ac1_conflict[i]]
        #                lat0_ac2 = self.lat[ac2_conflict[i]]
        #                lon0_ac1 = self.lon[ac1_conflict[i]]
        #                lon0_ac2 = self.lon[ac2_conflict[i]]
        #
        #                # DETAILED CPA
        ##                t_cpa_ac1ac2 = self.t_cpa[ac1_conflict[i]][ac2_conflict[i]]
        ##                t_cpa_range = np.arange(t_cpa_ac1ac2-range_t,t_cpa_ac1ac2+range_t,delta_t)
        ##                dist_ac1 = t_cpa_precise*self.spd[ac1_conflict[i]]*nm
        ##                dist_ac2 = t_cpa_precise*self.spd[ac2_conflict[i]]*nm
        ##
        ##
        ##                plat_ac1 = np.append(plat_ac1,lat0_ac1 + \
        ##                             np.degrees(dist_ac1*np.cos(np.radians(self.trk[ac1_conflict[i]]))/Rearth))
        ##                plat_ac2 = np.append(plat_ac2,lat0_ac2 + \
        ##                             np.degrees(dist_ac2*np.cos(np.radians(self.trk[ac2_conflict[i]]))/Rearth))
        ##                plon_ac1 = np.append(plon_ac1,lon0_ac1 + \
        ##                                   np.degrees(dist_ac1*np.sin(np.radians(self.trk[ac1_conflict[i]])) \
        ##                                     /np.cos(np.radians(lat0_ac1))/Rearth))
        ##                plon_ac2 = np.append(plon_ac2,lon0_ac2 + \
        ##                                   np.degrees(dist_ac2*np.sin(np.radians(self.trk[ac2_conflict[i]])) \
        ##                                       /np.cos(np.radians(lat0_ac2))/Rearth))
        ##
        ##                dist_ac1ac2 = np.array([])
        ##                for j in range(0,np.size(t_cpa_range)):
        ##                    dist_ac1ac2 = np.append(dist_ac1ac2,(latlondist(plat_ac1[j],plon_ac1[j],plat_ac2[j],plon_ac2[j]))/nm)
        ##
        ##                index_new_t_cpa = np.argmin(dist_ac1ac2)
        ##                self.dist_cpa[ac1_conflict[i]][ac2_conflict[i]] = dist_ac1ac2[index_new_t_cpa]
        ##
        ##                self.t_cpa[ac1_conflict[i]][ac2_conflict[i]] = t_cpa_precise[index_new_t_cpa]
        #
        #                if heading_range == 1:
        #                    self.heading_range(i,ac1_conflict,ac2_conflict,lat0_ac1,lon0_ac1,lat0_ac2,lon0_ac2)
        #
        #        return

    def apply_heading_range(self):
        Va = np.mat(self.spd)
        Ha = np.radians(self.trk)
        Ha_len = len(Ha)
        Ha = np.repeat(Ha,Ha_len)

        Ha = Ha.reshape(Ha_len,Ha_len)

        Vb = Va
        Hb = Ha

        R = self.dist_range

        [H0,S0] = geo.qdrdist_matrix(np.mat(self.lat),np.mat(self.lon),np.mat(self.lat),np.mat(self.lon))

        S0 = self.apply_before_filter(S0,Va)

        H0 = np.radians(H0.T)
        R_S0 = np.divide(R,S0)
        arcsin = np.arcsin(R_S0)
        Hr_new1 = H0 - arcsin
        Hr_new2 = H0 + arcsin
        Vb_Va = np.divide(Vb.T,Va)
        Hb_Hr_sin1 = np.sin(Hr_new1 - Hb)
        mp1 = np.multiply(Vb_Va,Hb_Hr_sin1)
        mp1 = (mp1 + 1)%2 - 1
        Hb_Hr_sin2 = np.sin(Hr_new2 - Hb)
        mp2 = np.multiply(Vb_Va,Hb_Hr_sin2)
        mp2 = (mp2 + 1)%2 - 1

        Ha_new11 = Hr_new1 - np.arcsin(mp1)
        Ha_new12 = Hr_new1 - np.arcsin(np.pi-mp1)
        Ha_new21 = Hr_new2 - np.arcsin(mp2)
        Ha_new22 = Hr_new2 - np.arcsin(np.pi-mp2)

        Ha_new11 = np.degrees(Ha_new11)
        Ha_new12 = np.degrees(Ha_new12)
        Ha_new21 = np.degrees(Ha_new21)
        Ha_new22 = np.degrees(Ha_new22)

        Ha_new11 = (Ha_new11+180)%360-180
        Ha_new12 = (Ha_new12+180)%360-180
        Ha_new21 = (Ha_new21+180)%360-180
        Ha_new22 = (Ha_new22+180)%360-180

        # Check for more than 90-degree headings, TBD
        Hdeg = np.degrees(Ha.T)
        Heading_1 = np.subtract(Hdeg,90)
        Heading_2 = np.add(Hdeg,90)

        # Heading_new11
        condition11a = Ha_new11 < Heading_1
        condition11b = S0>0
        condition11 = np.multiply(condition11a,condition11b)
        condition11 = np.invert(condition11)
        Ha_new11 = np.where(condition11,Ha_new11,Heading_1)

        # Heading_new21
        condition21a = Ha_new21 > Heading_2
        condition21 = np.multiply(condition21a,condition11b)
        condition21 = np.invert(condition21)
        Ha_new21 = np.where(condition21,Ha_new21,Heading_2)

        # Condition where Va>Vb and angle_rel does not include in H0_limits
        va_sxa = np.multiply(Va,np.sin(Ha))
        vb_sxb = va_sxa
        va_cxa = np.multiply(Va,np.cos(Ha))
        vb_cxb = va_cxa
        angle_rel = np.arctan2(np.subtract(va_sxa.T,vb_sxb),np.subtract(va_cxa.T,vb_cxb))
        angle_rel = np.degrees(angle_rel)
        angle_rel = (angle_rel+180)%360-180

        condition1 = Va.T > Vb
        H0_lim1 = np.degrees(H0) - 90
        H0_lim2 = np.degrees(H0) + 90
        H0_lim1 = (H0_lim1+180)%360-180
        H0_lim2 = (H0_lim2+180)%360-180
        condition2a = angle_rel < H0_lim1
        condition2b = angle_rel > H0_lim2

        condition2 = np.multiply(condition2a,condition2b)
        condition = np.multiply(condition1,condition2)
        condition = np.invert(condition)

        Ha_new11 = np.where(condition,Ha_new11,np.nan)
        Ha_new12 = np.where(condition,Ha_new12,np.nan)
        Ha_new21 = np.where(condition,Ha_new21,np.nan)
        Ha_new22 = np.where(condition,Ha_new22,np.nan)

        # S0_S0 = np.multiply(S0,S0)
        # d_sep = self.dist_range**2
        # aaa = np.multiply(2,np.multiply(S0,self.dist_range))
        # d_min = H0
        # condition = np.abs(np.subtract(H0,x_rel)) < np.radians(90)
        # d_min = np.where(condition,d_min,np.multiply(H0,np.sin(np.subtract(H0,x_rel))))
        # cos_b = np.arcsin(np.divide(d_min,self.dist_range)) - np.abs(np.subtract(H0,x_rel))
        # mp = np.multiply(aaa,cos_b)
        # upper = np.sqrt(S0_S0 + d_sep - mp)
        # Va_Va = np.multiply(Va,Va)
        # Vb_Vb = np.multiply(Vb,Vb)
        # va_vb = np.multiply(Va,Vb)
        # a_va_vb = np.multiply(2,va_vb)
        # mult = np.multiply(a_va_vb,np.cos(np.subtract(Ha.T,Hb)))
        # Va_Va_Vb_Vb = np.add(Va_Va,Vb_Vb)

        # v_rel = np.sqrt(np.subtract(Va_Va_Vb_Vb,mult))
        # print v_rel
        # t_fls = np.divide(upper,v_rel)
        # print t_fls

        diff_11_21 = np.subtract(Ha_new11,Ha_new21)
        diff_11_21 = np.abs((diff_11_21 + 180) % 360 - 180)

        diff_12_22 = np.subtract(Ha_new12,Ha_new22)
        diff_12_22 = np.abs((diff_12_22 + 180)  % 360 - 180)

        # now combine into 1 complexity score
        Compl_1 = np.divide(diff_11_21,360)
        Compl_2 = np.divide(diff_12_22,360)
        Compl_12ac = np.vstack((Compl_1,Compl_2))
        Compl_ac = np.nansum(Compl_12ac, axis = 0)

        # Condition where S0 < self.dist_range
        # condition = np.multiply(S0<self.dist_range,S0>0)
        # condition = np.invert(condition)
        # Compl_ac = np.where(condition,Compl_ac,0.5)

        Compl_ac = np.nansum(Compl_ac)

        self.complexity[self.step][0] = Compl_ac #/ self.ntraf
        self.complexity[self.step][1] = Compl_ac / self.ntraf

        self.complexity_plot()
        return


class Metric():
    """
    Metric class definition : traffic metrics

    Methods:
        Metric()                :  constructor

        update()                : add a command to the command stack
        close()                 : close file

    Created by  : Jacco M. Hoekstra (TU Delft)

    GitHub Metric, full collaborator MICHON-BS!!
    """

    def __init__(self):
        # Create metrics file
        # fname = settings.log_path + \
        #     strftime("%Y-%m-%d-%H-%M-%S-BlueSky-Metrics.txt", gmtime())
        # self.file = open(fname,"w")

        # Write header
#        self.write(0.0,"Header info tbd")

        # Last time for which Metrics.update was called
        self.t0 = -9999   # force first time call

        # Set time interval in seconds
        self.dt = 1  # [seconds]

        self.name = ("CoCa-Metric","HB-Metric","Delete AC")
        self.metric_number = -1
        self.fir_circle_point = [0.,0.]
        self.fir_circle_radius = 0.
        self.fir_number = 0
        self.metricstime = 0
        self.tbegin = 0

        self.metric_Area = metric_Area()

        self.cells = self.metric_Area.makeRegions()

        self.cellarea = self.metric_Area.cellArea()
        self.metric = (metric_CoCa(self.metric_Area),metric_HB(self.cellarea))

        return

    def toggle(self, flag, dt=None):
        """ Toggle metrics module from stack """
        if type(flag) == bool and not flag:
            # Can toggle metrics off with 'OFF'
            self.metric_number = -1
        else:
            # Otherwise select a metric to run
            if flag <= 0:
                return True, "Metrics OFF"

            if flag > len(self.name):
                return False, "No such metric"

            if not bs.traf.area == "Circle":
                return False, "First define AREA for metric"

            if dt:
                self.dt = dt

            self.metric_number = flag - 1

            return True, "Activated %s (%d), dt=%.2f" % (self.name[self.metric_number], flag, self.dt)

    def write(self,t,line):
        """
        Write: can be called from anywhere traf.metric.write( txt )
        Adds time stamp and ';'
        """
        self.file.write(tim2txt(t)+";"+line+chr(13)+chr(10))
        return

    def update(self):
        #check if configured and there is actual traffic
        if self.metric_number == -1 or bs.traf.ntraf < 1:
            return

        """Update: to call for regular logging & runtime analysis"""
        # Only do something when time is there
        if abs(bs.sim.simt-self.t0)<self.dt:
            return
        self.t0 = bs.sim.simt  # Update time for scheduler
        if self.metricstime == 0:
            self.tbegin = bs.sim.simt
            self.metricstime = 1
            print("METRICS STARTED")
            # FIR_circle(bs.navdb,self.fir_number)
            # cmd.stack("AREA "+str(self.cellarea[2][0])+","+str(self.cellarea[2][1])+ \
            #   ","+str(self.cellarea[0][0])+","+str(self.cellarea[0][1]))

        # A lot of smart Michon-code here, probably using numpy arrays etc.
        if bs.sim.simt >= 0:
            if self.metric_number == 0:
                self.metric[self.metric_number].AircraftCell(self.cells,bs.sim.t-self.tbegin)
            elif self.metric_number == 1:
                self.metric[self.metric_number].applymetric()

        print("Number of Aircraft in Research Area (FIR):" + str(self.metric[self.metric_number].ntraf))

        deleteAC = []
        for i in range(0,bs.traf.ntraf):
            if bs.traf.selvs[i] <= 0 and (bs.traf.selalt[i]/ft) < 750 and bs.traf.selspd[i] < 300:
                deleteAC.append(bs.traf.id[i])

            elif bs.traf.selvs[i] <=0 and (bs.traf.selalt[i]/ft) < 10.:
                deleteAC.append(bs.traf.id[i])

            if bs.traf.selvs[i] <=0 and bs.traf.selspd[i] < 10.:
                deleteAC.append(bs.traf.id[i])

        for i in range(0,len(deleteAC)):
            bs.traf.delete(deleteAC[i])

        # Heartbeat for test
        self.write(bs.sim.simt,"NTRAF;"+str(bs.traf.ntraf))
        return

    def plot(self):
        # Pause simulation
        bs.sim.hold()

        # Open a plot window attached to a command?
        #    plot, showplot and other matplotlib commands

        # Continue simulation
        bs.sim.op()
        return
