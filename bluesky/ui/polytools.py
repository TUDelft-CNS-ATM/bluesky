""" Polygon functions """
import numpy as np
import OpenGL.GLU as glu


class PolygonSet:
    """ Generate one or more polygons from their contours. """
    in_poly = False
    in_contour = False
    tess = None

    def __init__(self):
        self.vbuf = []
        self.prevnode = None
        self.prevcp = None
        self.start_vertex = None
        self.start_cp = None

        if not PolygonSet.tess:
            PolygonSet.tess = glu.gluNewTess()
            glu.gluTessCallback(PolygonSet.tess, glu.GLU_TESS_VERTEX_DATA, lambda vertex, vbuf: vbuf.extend(vertex[0:2]))
            glu.gluTessCallback(PolygonSet.tess, glu.GLU_EDGE_FLAG, lambda flag: None)
            glu.gluTessCallback(PolygonSet.tess, glu.GLU_TESS_COMBINE, lambda c, d, w: np.array(c))

    def bufsize(self):
        return len(self.vbuf)

    def begin(self):
        self.end()
        glu.gluTessBeginPolygon(PolygonSet.tess, self.vbuf)
        PolygonSet.in_poly = True

    def end(self):
        if PolygonSet.in_poly:
            self.endContour()
            glu.gluEndPolygon(PolygonSet.tess)
            PolygonSet.in_poly = False

    def beginContour(self):
        self.endContour()
        if not PolygonSet.in_poly:
            self.begin()
        glu.gluTessBeginContour(PolygonSet.tess)
        PolygonSet.in_contour = True

    def endContour(self):
        if PolygonSet.in_contour:
            if self.prevcp is not None or self.start_cp is not None:
                self.addVertex(self.start_vertex, self.start_cp)
            glu.gluTessEndContour(PolygonSet.tess)
            PolygonSet.in_contour = False
            self.prevcp = None
            self.prevnode = None
            self.start_vertex = None
            self.start_cp = None

    def addContour(self, contour):
        self.begin()
        for vertex in np.reshape(contour, (len(contour) // 2, 2)):
            self.addVertex(np.append(vertex, 0.0))
        self.end()

    def addVertex(self, vertex, controlpoint=None):
        if not PolygonSet.in_contour:
            self.beginContour()
            self.start_vertex = vertex
            self.start_cp = controlpoint
            glu.gluTessVertex(PolygonSet.tess, vertex, vertex)

        elif abs(vertex[0] - self.prevnode[0]) >= 1e-7 or abs(vertex[1] - self.prevnode[1]) >= 1e-7:
            if (controlpoint is None and self.prevcp is None):
                glu.gluTessVertex(PolygonSet.tess, vertex, vertex)
            else:
                if controlpoint is not None:
                    if self.prevcp is not None:
                        self.bezier2(vertex, 2 * vertex - controlpoint)
                    else:
                        self.bezier1(vertex, 2 * vertex - controlpoint)

                else:
                    self.bezier1(vertex, self.prevcp)

        self.prevnode = vertex
        self.prevcp = controlpoint

    def bezier1(self, vertex, controlpoint):
        for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
            lnode1 = self.prevnode + fraction * (controlpoint - self.prevnode)
            lnode2 = controlpoint + fraction * (vertex - controlpoint)

            vnew = lnode1 + fraction * (lnode2 - lnode1)
            glu.gluTessVertex(PolygonSet.tess, vnew, vnew)

    def bezier2(self, vertex, controlpoint):
        for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
            lnode1 = self.prevnode + fraction * (self.prevcp - self.prevnode)
            lnode2 = self.prevcp + fraction * (controlpoint - self.prevcp)
            lnode3 = controlpoint + fraction * (vertex - controlpoint)

            lnode4 = lnode1 + fraction * (lnode2 - lnode1)
            lnode5 = lnode2 + fraction * (lnode3 - lnode2)

            vnew = lnode4 + fraction * (lnode5 - lnode4)
            glu.gluTessVertex(PolygonSet.tess, vnew, vnew)


class BoundingBox:
    """ Calculate bounding box for a set of vertices. """

    def __init__(self):
        self.corners = [999.9, -999.9, 999.9, -999.9]

    def update(self, vertex):
        self.corners[0] = min(self.corners[0], vertex[0])
        self.corners[1] = max(self.corners[1], vertex[0])
        self.corners[2] = min(self.corners[2], vertex[1])
        self.corners[3] = max(self.corners[3], vertex[1])

    def center(self):
        return [0.5 * (self.corners[0] + self.corners[1]),
                0.5 * (self.corners[2] + self.corners[3])]
