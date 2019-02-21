''' Load visual data from text files.'''
import os
from math import cos, radians, degrees, sqrt, atan2, sin, asin
from zipfile import ZipFile
import numpy as np
import bluesky as bs
from bluesky import settings

## Default settings
settings.set_variable_defaults(navdata_path='data/navdata')

REARTH_INV = 1.56961231e-7


def load_coastline_txt():
    # -------------------------COASTLINE DATA----------------------------------
    # Init geo (coastline)  data and convert pen up/pen down format of
    # coastlines to numpy arrays with lat/lon
    coast = []
    clat = clon = 0.0
    with open(os.path.join(settings.navdata_path, 'coastlines.dat'), 'r') as f:
        print("Reading coastlines.dat")
        for line in f:
            line = line.strip()
            if not (line == "" or line[0] == '#'):
                arg = line.split()
                if len(arg) == 3:
                    lat, lon = float(arg[1]), float(arg[2])
                    if arg[0] == 'D':
                        coast.append([clat, clon, lat, lon])
                    clat, clon = lat, lon
    # Sort the line segments by longitude of the first vertex
    coastvertices = np.array(
        sorted(coast, key=lambda a_entry: a_entry[1]), dtype=np.float32)
    coastindices = np.zeros(361)
    coastlon = coastvertices[:, 1]
    for i in range(0, 360):
        coastindices[i] = np.searchsorted(coastlon, i - 180) * 2
    coastvertices.resize((int(coastvertices.size / 2), 2))
    del coast
    return coastvertices, coastindices


# Only try this if BlueSky is started in qtgl gui mode
if bs.gui_type == 'qtgl':
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QProgressDialog
    from bluesky.ui.polytools import PolygonSet, BoundingBox


    class ProgressBar():
        def __init__(self, text):
            if QApplication.instance() is None:
                self.dialog = None
                print(text)
            else:
                self.dialog = QProgressDialog(text, 'Cancel', 0, 100)
                self.dialog.setWindowFlags(Qt.WindowStaysOnTopHint)
                self.dialog.show()

        def update(self, value):
            if self.dialog:
                self.dialog.setValue(value)
                QApplication.processEvents()
            else:
                print('Progress: %.2f%% done' % value)

        def close(self):
            if self.dialog:
                self.dialog.close()


    """ Load static data for GUI from files such as airport, shapes, etc."""

    def load_aptsurface_txt():
        """ Read airport data from navigation database files"""
        pb = ProgressBar('Binary buffer file not found, or file out of date: Constructing vertex buffers from geo data.')

        runways = []
        rwythr = []
        asphalt = PolygonSet()
        concrete = PolygonSet()
        cur_poly = asphalt
        apt_indices = []
        apt_ctr_lat = []
        apt_ctr_lon = []
        apt_bb = BoundingBox()
        count = 0
        bytecount = 0
        zfile = ZipFile(os.path.join(settings.navdata_path, 'apt.zip'))
        fsize = float(zfile.getinfo('apt.dat').file_size)
        print("Reading apt.dat from apt.zip")
        with zfile.open('apt.dat', 'r') as f:
            for line in f:
                bytecount += len(line)
                # Check how far we are
                count += 1
                if count % 1000 == 0:
                    pb.update((bytecount / fsize * 100.0))

                elems = line.decode(encoding="ascii", errors="ignore").strip().split()
                if not elems:
                    continue

                # 1: AIRPORT
                if elems[0] in ['1', '16', '17']: # 1=airport, 16=seaplane base, 17=heliport
                    cur_poly.end()
                    # Store the starting vertices for this apt, [0, 0] if this is the first apt
                    if apt_indices:
                        start_indices = [apt_indices[-1][0] + apt_indices[-1][1],
                                         apt_indices[-1][2] + apt_indices[-1][3]]
                    else:
                        start_indices = [0, 0]

                    if asphalt.bufsize() // 2 > start_indices[0] or concrete.bufsize() // 2 > start_indices[1]:
                        apt_indices.append( [
                                                start_indices[0],
                                                asphalt.bufsize() // 2 - start_indices[0],
                                                start_indices[1],
                                                concrete.bufsize() // 2 - start_indices[1]
                                            ])

                        center = apt_bb.center()
                        apt_ctr_lat.append(center[0])
                        apt_ctr_lon.append(center[1])

                    # Reset the boundingbox
                    apt_bb = BoundingBox()
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
                    offset0 = float(elems[11])

                    lat1 = float(elems[18])
                    lon1 = float(elems[19])
                    offset1 = float(elems[20])

                    # runway vertices
                    runways.extend(dlatlon(lat0, lon0, lat1, lon1, width))

                    # threshold information: ICAO code airport, Runway identifier,
                    # latitude, longitude, bearing
                    # vertices: gives vertices of the box around the threshold

                    # opposite runways are on the same line. RWY1: 8-11, RWY2: 17-20
                    # Hence, there are two thresholds per line
                    # thr0: First lat0 and lon0 , then lat1 and lat1, offset=[11]
                    # thr1: First lat1 and lat1 , then lat0 and lon0, offset=[20]

                    thr0, vertices0 = thresholds(radians(lat0), radians(lon0), radians(lat1), radians(lon1), offset0)
                    thr1, vertices1 = thresholds(radians(lat1), radians(lon1), radians(lat0), radians(lon0), offset1)
                    rwythr.extend(vertices0)
                    rwythr.extend(vertices1)

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
                    continue

                controlpoint = None
                # Straight line (111) or bezier line (112) in contour
                if elems[0] == '111' or elems[0] == '112':
                    vertex = np.array((float(elems[1]), float(elems[2]), 0.0))
                    apt_bb.update(vertex)

                    if cur_poly.in_poly:
                        if elems[0] == '112':
                            controlpoint = np.array((float(elems[3]), float(elems[4]), 0.0))

                        cur_poly.addVertex(vertex, controlpoint)

                # Straight (113) or bezier (114) contour endpoint
                elif elems[0] == '113' or elems[0] == '114':
                    vertex = np.array((float(elems[1]), float(elems[2]), 0.0))
                    apt_bb.update(vertex)

                    if cur_poly.in_poly:
                        if elems[0] == '114':
                            controlpoint = np.array((float(elems[3]), float(elems[4]), 0.0))

                        cur_poly.addVertex(vertex, controlpoint)

                        cur_poly.endContour()

                else:
                    cur_poly.end()

        # calculate the location of the runway thresholds

        # Clean up:
        cur_poly.end()
        start_indices = [apt_indices[-1][0] + apt_indices[-1][1],
                         apt_indices[-1][2] + apt_indices[-1][3]]

        if asphalt.bufsize() > start_indices[0] or concrete.bufsize() > start_indices[1]:
            apt_indices.append( [
                                    start_indices[0],
                                    int(asphalt.bufsize() / 2) - start_indices[0],
                                    start_indices[1],
                                    int(concrete.bufsize() / 2) - start_indices[1]
                                ])

            center = apt_bb.center()
            apt_ctr_lat.append(center[0])
            apt_ctr_lon.append(center[1])

        # Store in binary pickle file
        vbuf_asphalt = np.array(asphalt.vbuf, dtype=np.float32)
        vbuf_concrete = np.array(concrete.vbuf, dtype=np.float32)
        vbuf_runways = np.array(runways, dtype=np.float32)
        vbuf_rwythr = np.array(rwythr, dtype=np.float32)
        apt_ctr_lat = np.array(apt_ctr_lat)
        apt_ctr_lon = np.array(apt_ctr_lon)
        apt_indices = np.array(apt_indices)

        # Close the progress dialog
        pb.close()

        # return the data
        return vbuf_asphalt, vbuf_concrete, vbuf_runways, vbuf_rwythr, apt_ctr_lat, apt_ctr_lon, apt_indices


# Runway threshold loader for navdatabase
def navdata_load_rwythresholds():
    rwythresholds = dict()
    curthresholds = None
    zfile = ZipFile(os.path.join(settings.navdata_path, 'apt.zip'))
    print("Reading apt.dat from apt.zip")
    with zfile.open('apt.dat', 'r') as f:
        for line in f:
            elems = line.decode(encoding="ascii", errors="ignore").strip().split()
            if len(elems) == 0:
                continue

            # 1: AIRPORT
            if elems[0] == '1':
                # Add airport to runway threshold database
                curthresholds = dict()
                rwythresholds[elems[4]] = curthresholds
                continue

            if elems[0] == '100':
                # Only asphalt and concrete runways
                if int(elems[2]) > 2:
                    continue
                # rwy_lbl = (elems[8], elems[17])

                lat0    = float(elems[9])
                lon0    = float(elems[10])
                offset0 = float(elems[11])

                lat1    = float(elems[18])
                lon1    = float(elems[19])
                offset1 = float(elems[20])

                # threshold information: ICAO code airport, Runway identifier,
                # latitude, longitude, bearing
                # vertices: gives vertices of the box around the threshold

                # opposite runways are on the same line. RWY1: 8-11, RWY2: 17-20
                # Hence, there are two thresholds per line
                # thr0: First lat0 and lon0 , then lat1 and lat1, offset=[11]
                # thr1: First lat1 and lat1 , then lat0 and lon0, offset=[20]

                thr0, _ = thresholds(radians(lat0), radians(lon0), radians(lat1), radians(lon1), offset0)
                thr1, _ = thresholds(radians(lat1), radians(lon1), radians(lat0), radians(lon0), offset1)
                curthresholds[elems[8]]  = thr0
                curthresholds[elems[17]] = thr1
                continue
    return rwythresholds


# calculates the threshold points per runway
# underlying equations can be found at
# http://www.movable-type.co.uk/scripts/latlong.html
def thresholds(lat1, lon1, lat2, lon2, offset):
    d = offset * REARTH_INV
    d_box = 20.0 * REARTH_INV  # m
    width_box = 30  # m
    deltal = lon2 - lon1

# calculate runway bearing
    bearing = atan2(sin(deltal) * cos(lat2), (cos(lat1) * sin(lat2) -
                                              sin(lat1) * cos(lat2) * cos(deltal)))

    # normalize to 0-360 degrees
    bearing = radians((degrees(bearing) + 360) % 360)

    # bearing in opposite direction
    opp_bearing = radians((degrees(bearing) + 180) % 360)

    # get threshold points
    latthres, lonthres = thrpoints(lat1, lon1, d, bearing)

    # get points at both ends of the boxes around the threshold point
    # (zebra crossing)
    latbox0, lonbox0 = thrpoints(latthres, lonthres, d_box, bearing)
    latbox1, lonbox1 = thrpoints(latthres, lonthres, d_box, opp_bearing)

    # calculate vertices of threshold box
    vertices = dlatlon(degrees(latbox0), degrees(lonbox0), degrees(latbox1),
                       degrees(lonbox1), width_box)

    return (degrees(latthres), degrees(lonthres), degrees(bearing)), vertices


# calculate threshold points as well as end points of threshold box
# underlying equations can be found at
# http://www.movable-type.co.uk/scripts/latlong.html

def thrpoints(lat1, lon1, d, bearing):

    latthres = asin(sin(lat1) * cos(d) + cos(lat1) * sin(d) * cos(bearing))

    lonthres = lon1 + atan2(sin(bearing) * sin(d) * cos(lat1),
                            cos(d) - sin(lat1) * sin(latthres))

    return latthres, lonthres


# used for calculating the vertices of the runways as well as the threshold boxes
def dlatlon(lat0, lon0, lat1, lon1, width):

    # calculate distance between ends of runways / threshold boxes
    flat_earth = cos(0.5 * radians(lat0 + lat1))
    lx = lat1 - lat0
    ly = (lon1 - lon0) * flat_earth
    l = sqrt(lx * lx + ly * ly)
    wx = ly / l * 0.5 * width
    wy = -lx / l * 0.5 * width
    dlat = degrees(wx * REARTH_INV)
    dlon = degrees(wy * REARTH_INV / flat_earth)

    # store the vertice information per runway /threshold box
    vertices = [lat0 + dlat, lon0 + dlon,
                lat0 - dlat, lon0 - dlon,
                lat1 + dlat, lon1 + dlon,
                lat0 - dlat, lon0 - dlon,
                lat1 + dlat, lon1 + dlon,
                lat1 - dlat, lon1 - dlon]

    return vertices
