"""Area filter module"""
from geo import kwikdist_matrix
from matplotlib.path import Path
import numpy as np


areas = dict()


def defineArea(scr, areaname, areatype, coordinates):
    # When top is skipped in stack, None is entered instead. Replace with 1e9
    if coordinates[-2] is None:
        coordinates[-2] = 1e9

    if areatype == 'BOX':
        areas[areaname] = Box(coordinates[:4], *coordinates[4:])
    elif areatype == 'CIRCLE':
        areas[areaname] = Circle(coordinates[:2], *coordinates[2:])
    elif areatype == 'POLY':
        areas[areaname] = Poly(coordinates)
    elif areatype == 'POLYALT':
        areas[areaname] = Poly(coordinates[2:], *coordinates[:2])

    # Pass the shape on to the screen object
    scr.objappend(areatype, areaname, coordinates)
    
def checkInside(areaname, lat, lon, alt):
    '''Returns an array of booleans. True ==  Inside'''
    if areaname not in areas:
        return []
    area = areas[areaname]
    return area.checkInside(lat, lon, alt)

def deleteArea(scr, areaname):
    if areaname in areas:
        areas.pop(areaname)
        scr.objappend('', areaname, None)    

def reset():
    areas.clear()


class Box:
    def __init__(self, coordinates, top=1e9, bottom=-1e9):
        self.top    = top
        self.bottom = bottom        
        # Sort the order of the corner points 
        self.lat0 = min(coordinates[0],coordinates[2])
        self.lon0 = min(coordinates[1],coordinates[3])
        self.lat1 = max(coordinates[0],coordinates[2])
        self.lon1 = max(coordinates[1],coordinates[3])
        
    def checkInside(self, lat, lon, alt):     
        inside = ((self.lat0 <=  lat) & ( lat <= self.lat1)) & \
                 ((self.lon0 <= lon) & (lon <= self.lon1)) & \
                 ((self.bottom <= alt) & (alt <= self.top))      
        return inside
        


class Circle:
    def __init__(self, center, radius, top=1e9, bottom=-1e9):
        self.clat   = center[0]
        self.clon   = center[1]
        self.r      = radius
        self.top    = top
        self.bottom = bottom
         
    def checkInside(self, lat, lon, alt): 
        clat     = np.array([self.clat]*len( lat))
        clon     = np.array([self.clon]*len( lat))
        r        = np.array([self.r]*len( lat))        
        distance = kwikdist_matrix(clat, clon,  lat, lon)  # [NM]        
        inside   = (distance <= r) & (self.bottom <= alt) & (alt <= self.top)
        return inside
        


class Poly:
    def __init__(self, coordinates, top=1e9, bottom=-1e9):
        self.border = Path(np.reshape(coordinates, (len(coordinates) / 2, 2)))
        self.top    = top
        self.bottom = bottom
            
    def checkInside(self, lat, lon, alt):
        points = np.vstack((lat,lon)).T         
        inside = self.border.contains_points(points) & (self.bottom <= alt) & (alt <= self.top)        
        return inside
    