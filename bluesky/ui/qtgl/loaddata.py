try:
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QApplication, QProgressDialog
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QProgressDialog

import cPickle as pickle
import OpenGL.GLU as glu
import numpy as np
import os
from math import cos, radians, degrees, sqrt
from zipfile import ZipFile

tess = glu.gluNewTess()
glu.gluTessCallback(tess, glu.GLU_TESS_VERTEX_DATA, lambda vertex, vbuf: vbuf.extend(vertex[0:2]))
glu.gluTessCallback(tess, glu.GLU_EDGE_FLAG, lambda flag: None)
glu.gluTessCallback(tess, glu.GLU_TESS_COMBINE, lambda c, d, w: np.array(c))


class PolygonSet:
    in_poly    = False
    in_contour = False

    def __init__(self):
        self.vbuf           = []
        self.prevnode       = None
        self.prevcp         = None
        self.start_vertex   = None
        self.start_cp       = None

    def bufsize(self):
        return len(self.vbuf)

    def begin(self):
        self.end()
        glu.gluTessBeginPolygon(tess, self.vbuf)
        PolygonSet.in_poly = True

    def end(self):
        if PolygonSet.in_poly:
            self.endContour()
            glu.gluEndPolygon(tess)
            PolygonSet.in_poly = False

    def beginContour(self):
        self.endContour()
        if not PolygonSet.in_poly:
            self.begin()
        glu.gluTessBeginContour(tess)
        PolygonSet.in_contour = True

    def endContour(self):
        if PolygonSet.in_contour:
            if self.prevcp is not None or self.start_cp is not None:
                self.addVertex(self.start_vertex, self.start_cp)
            glu.gluTessEndContour(tess)
            PolygonSet.in_contour = False
            self.prevcp           = None
            self.prevnode         = None
            self.start_vertex     = None
            self.start_cp         = None

    def addVertex(self, vertex, controlpoint=None):
        if not PolygonSet.in_contour:
            self.beginContour()
            self.start_vertex = vertex
            self.start_cp     = controlpoint
            glu.gluTessVertex(tess, vertex, vertex)

        elif abs(vertex[0] - self.prevnode[0]) >= 1e-7 or abs(vertex[1] - self.prevnode[1]) >= 1e-7:
            if (controlpoint is None and self.prevcp is None):
                glu.gluTessVertex(tess, vertex, vertex)
            else:
                if controlpoint is not None:
                    if self.prevcp is not None:
                        self.bezier2(vertex, 2 * vertex - controlpoint)
                    else:
                        self.bezier1(vertex, 2 * vertex - controlpoint)

                else:
                    self.bezier1(vertex, self.prevcp)

        self.prevnode = vertex
        self.prevcp   = controlpoint

    def bezier1(self, vertex, controlpoint):
        for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
            lnode1 = self.prevnode + fraction * (controlpoint - self.prevnode)
            lnode2 = controlpoint  + fraction * (vertex - controlpoint)

            vnew   = lnode1 + fraction * (lnode2 - lnode1)
            glu.gluTessVertex(tess, vnew, vnew)

    def bezier2(self, vertex, controlpoint):
        for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
            lnode1 = self.prevnode + fraction * (self.prevcp  - self.prevnode)
            lnode2 = self.prevcp   + fraction * (controlpoint - self.prevcp)
            lnode3 = controlpoint  + fraction * (vertex       - controlpoint)

            lnode4 = lnode1 + fraction * (lnode2 - lnode1)
            lnode5 = lnode2 + fraction * (lnode3 - lnode2)

            vnew   = lnode4 + fraction * (lnode5 - lnode4)
            glu.gluTessVertex(tess, vnew, vnew)


class BoundingBox:
    def __init__(self):
        self.corners = [999.9, -999.9, 999.9, -999.9]

    def update(self, vertex):
        self.corners[0] = min(self.corners[0], vertex[0])
        self.corners[1] = max(self.corners[1], vertex[0])
        self.corners[2] = min(self.corners[2], vertex[1])
        self.corners[3] = max(self.corners[3], vertex[1])

    def center(self):
        return [0.5 * (self.corners[0] + self.corners[1]), 0.5 * (self.corners[2] + self.corners[3])]


def load_airport_data():
    if os.path.isfile('data/global/apt.p'):
        f_vdata       = open('data/global/apt.p', 'rb')
        vbuf_asphalt  = pickle.load(f_vdata)
        vbuf_concrete = pickle.load(f_vdata)
        vbuf_runways  = pickle.load(f_vdata)
        apt_ctr_lat   = pickle.load(f_vdata)
        apt_ctr_lon   = pickle.load(f_vdata)
        apt_indices   = pickle.load(f_vdata)
        f_vdata.close()
    else:
        pb = QProgressDialog('Binary buffer file not found, or file out of date: Constructing vertex buffers from geo data.', 'Cancel', 0, 100)
        pb.setWindowFlags(Qt.WindowStaysOnTopHint)
        pb.show()

        REARTH_INV    = 1.56961231e-7
        runways       = []
        asphalt       = PolygonSet()
        concrete      = PolygonSet()
        cur_poly      = asphalt
        apt_indices   = []
        apt_ctr_lat   = []
        apt_ctr_lon   = []
        apt_bb        = None
        count         = 0
        bytecount     = 0
        #fsize         = os.stat('data/global/apt.dat').st_size
        zfile = ZipFile('data/global/apt.zip')
        fsize = float(zfile.getinfo('apt.dat').file_size)
        with zfile.open('apt.dat', 'r') as f:
            for line in f:
                bytecount += len(line)
                # Check how far we are
                count += 1
                if count % 1000 == 0:
                    pb.setValue((bytecount / fsize * 100.0))
                    QApplication.processEvents()

                elems = line.strip().split()
                if len(elems) == 0:
                    continue

                # 1: AIRPORT
                if elems[0] == '1':
                    cur_poly.end()
                    if len(apt_indices) > 0:
                        # Store the number of vertices per airport
                        apt_indices[-1][1] = asphalt.bufsize()  / 2 - apt_indices[-1][0]
                        apt_indices[-1][3] = concrete.bufsize() / 2 - apt_indices[-1][2]

                    # Store the starting vertex index for the next airport
                    apt_indices.append([asphalt.bufsize() / 2, 0, concrete.bufsize() / 2, 0])
                    apt_ctr_lat.append(0.0)
                    apt_ctr_lon.append(0.0)
                    continue

                # 100: LAND RUNWAY
                if elems[0] == '100':
                    width = float(elems[1])
                    # Only asphalt and concrete runways
                    if int(elems[2]) > 2:
                        continue
                    # rwy_lbl = (elems[8], elems[17])
                    lat0 = float(elems[9])
                    lon0 = float(elems[10])
                    lat1 = float(elems[18])
                    lon1 = float(elems[19])
                    flat_earth = cos(0.5 * radians(lat0 + lat1))
                    lx = lat1 - lat0
                    ly = (lon1 - lon0) * flat_earth
                    l  = sqrt(lx * lx + ly * ly)
                    wx =  ly / l * 0.5 * width
                    wy = -lx / l * 0.5 * width
                    dlat = degrees(wx * REARTH_INV)
                    dlon = degrees(wy * REARTH_INV / flat_earth)
                    runways.extend([ lat0 + dlat, lon0 + dlon,
                                    lat0 - dlat, lon0 - dlon,
                                    lat1 + dlat, lon1 + dlon,
                                    lat0 - dlat, lon0 - dlon,
                                    lat1 + dlat, lon1 + dlon,
                                    lat1 - dlat, lon1 - dlon])
                    continue

                # 110: TAXIWAY/PAVEMENT: Start of polygon contour
                if elems[0] == '110':
                    cur_poly.end()
                    if elems[1] == '1':
                        cur_poly = asphalt
                        cur_poly.begin()
                    elif elems[1] == '2':
                        cur_poly = concrete
                        cur_poly.begin()
                    continue

                # 130: AIRPORT BOUNDARY
                if elems[0] == '130':
                    cur_poly.end()
                    apt_bb      = BoundingBox()
                    continue

                if cur_poly.in_poly is False and apt_bb is None:
                    continue

                controlpoint = None
                # Straight line (111) or bezier line (112) in contour
                if elems[0] == '111' or elems[0] == '112':
                    vertex           = np.array((float(elems[1]), float(elems[2]), 0.0))
                    if apt_bb is not None:
                        apt_bb.update(vertex)
                        continue
                    if elems[0] == '112':
                        controlpoint = np.array((float(elems[3]), float(elems[4]), 0.0))

                    cur_poly.addVertex(vertex, controlpoint)

                # Straight (113) or bezier (114) contour endpoint
                elif elems[0] == '113' or elems[0] == '114':
                    vertex           = np.array((float(elems[1]), float(elems[2]), 0.0))
                    if apt_bb is not None:
                        apt_bb.update(vertex)
                        apt_ctr_lat[-1], apt_ctr_lon[-1] = apt_bb.center()
                        apt_bb = None
                        continue
                    if elems[0] == '114':
                        controlpoint = np.array((float(elems[3]), float(elems[4]), 0.0))

                    cur_poly.addVertex(vertex, controlpoint)

                    cur_poly.endContour()

                else:
                    cur_poly.end()

        # Clean up:
        cur_poly.end()
        apt_indices[-1][1] = asphalt.bufsize()  / 2 - apt_indices[-1][0]
        apt_indices[-1][3] = concrete.bufsize() / 2 - apt_indices[-1][2]

        # Store in binary pickle file
        vbuf_asphalt  = np.array(asphalt.vbuf, dtype=np.float32)
        vbuf_concrete = np.array(concrete.vbuf, dtype=np.float32)
        vbuf_runways  = np.array(runways, dtype=np.float32)
        apt_ctr_lat   = np.array(apt_ctr_lat)
        apt_ctr_lon   = np.array(apt_ctr_lon)
        apt_indices   = np.array(apt_indices)

        f = open('data/global/apt.p', 'wb')
        pickle.dump(vbuf_asphalt, f, 2)
        pickle.dump(vbuf_concrete, f, 2)
        pickle.dump(vbuf_runways , f, 2)
        pickle.dump(apt_ctr_lat  , f, 2)
        pickle.dump(apt_ctr_lon  , f, 2)
        pickle.dump(apt_indices  , f, 2)
        f.close()

        # Close the progress dialog
        pb.close()

    # return the data
    return vbuf_asphalt, vbuf_concrete, vbuf_runways, apt_ctr_lat, apt_ctr_lon, apt_indices
