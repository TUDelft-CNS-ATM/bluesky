from math import *

import pygame as pg

import datetime, os
import os.path
import subprocess

import numpy as np

import bluesky as bs
from bluesky.tools import geo, areafilter
from bluesky.tools.aero import ft, kts, nm
from bluesky.tools.misc import tim2txt
from . import splash
from .keyboard import Keyboard
from .fastfont import Fastfont
from .console import Console
from .menu import Menu
from .dialog import fileopen as opendialog

black    = (0, 0, 0)
white    = (255,255,255)
darkgrey = (25, 25, 48)
grey     = (84, 84, 114)
darkblue = (25, 25, 64, 100)
white    = (255, 255, 255)
green    = (0, 255, 0)
blue     = (0, 0, 255)
red      = (255, 0, 0)
cyan     = (0,150,150)
lightgreyblue = (130, 150, 190)  # waypoint symbol color
lightgreygreen = (149, 215, 179)  # grid color
lightcyan = (0, 255, 255)  # FIR boundaries
amber    = (255,163,71)  # Conflicting aircraft
magenta  = (255,0,255) # Used for route

class Screen:
    """
    Screen class definition : contains radar & edit screen data & methods

    Methods:
        Screen(tmx)         :  constructor

        echo(msg)           : print something at screen
        update()            : Draw a new frame of screen
        ll2xy(lat,lon)      : lat/lon[deg] to pixel coordinate conversion
        xy2ll(x,y)          : pixel to lat/lon[de]g conversion
        zoom(factor)        : zoom in/out
        pan(lat,lon,txt)    : pan to lat,lon position

    Members: see constructor

    Created by : Jacco M. Hoekstra (TU Delft)
    Updated by : Jerom Maas

    """
    def __init__(self):
        # processes input from keyboard & mouse
        self.keyb = Keyboard()

        # Parameters for making screenshots
        self.session = "new"
        self.folder  = ""
        self.screenshot = False
        self.screenshotname = ""

        # Isometric display parameter
        self.isoalt = 0.  # how many meters one pixel is high

        # Display ADS-B range flag
        self.swAdsbCoverage = False

        # Update rate radar:
        self.radardt = 0.10  # 10x per sec 0.25  # 4x per second max
        self.radt0 = -999.  # last time drawn
        self.maxnrwp = 1000  # max nr apts+wpts to be drawn

    def init(self):
        # Read Screen configuration file:
        print()
        print("Setting up screen...")

        lst = np.genfromtxt("data/graphics/scr_cfg.dat", comments='#', dtype='i4')

        self.swfullscreen = int(lst[0]) == 0

        self.width = int(lst[1])  # default value to create variable
        self.height = int(lst[2])  # default value to create variable

        # Dimensions radar window
        self.lat1 = 53.  # [deg] upper limit display
        self.lat0 = 51.  # [deg] lowerlimit display
        self.lon0 = -1.  # [deg] left side of screen

        dellat = self.lat1 - self.lat0

        avelat = (self.lat0 + self.lat1) * 0.5

        dellon = dellat * self.width / (self.height * cos(radians(avelat)))
        avelon = (self.lon0 + dellon / 2. + 180.) % 360. - 180.

        self.lon1 = (self.lon0 + dellon + 180.) % 360. - 180.

        self.ctrlat = avelat
        self.ctrlon = avelon


        #----------------------------SYMBOLS-----------------------------
        # Read graphics for acsymbol (normal = green) + amber
        self.acsymbol = []
        for i in range(60):
            self.acsymbol.append(pg.image.load("data/graphics/acsymbol/acsymbol" \
                                  + str(i) + ".png"))

        self.ambacsymbol = []
        for i in range(60):
            self.ambacsymbol.append(pg.image.load("data/graphics/amb-acsymbol/amb-acsymbol" \
                                     + str(i) + ".png"))

        # Lable lines& default no trails
        self.swlabel = 3


        # Read and scale waypoint symbol
        wptgif = pg.image.load("data/graphics/waypoint.gif")
        self.wptsymbol = pg.transform.scale(wptgif, (10, 7))
        self.wpsw = 1  # 0=None, 1 = VOR 2 = non-digit ones, 3 =all

        # Read and scale airport symbol
        aptgif = pg.image.load("data/graphics/airport.gif")
        self.aptsymbol = pg.transform.scale(aptgif, (12, 9))
        self.apsw = 1  # 0 = None, 1 = Large, 2 = All

        # Free flight displays
        self.swsep = False # To show circles of 2.5 nm radius around each aircraft
                           # Note: circles will be distorted when away from equator
        self.swspd = False # To show speed vectors of each aircraft
        self.swtestarea= False

        #--------------------------------MAPS---------------------------------
        # Read map of world
        self.mapbmp = pg.image.load("data/graphics/world.jpg")
        w, h = self.mapbmp.get_size()

        # Two ref positions for scaling, convert to scaling factors x=a*lat+b
        x1, y1, lat1, lon1 = 0., 0., 90., -180.
        x2, y2, lat2, lon2 = w, h, -90., 180.
        self.mapax = (x2 - x1) / (lon2 - lon1)
        self.mapbx = x2 - self.mapax * lon2
        self.mapay = (y2 - y1) / (lat2 - lat1)
        self.mapby = y2 - self.mapay * lat2

        self.swsat = True

        # Nav display projection mode
        self.swnavdisp = False
        self.ndacid = ""
        self.ndlat = 0.0
        self.ndlon = 0.0
        self.ndhdg = 0.0

        #------------------------WINDOW SETUP and scaling--------------------------
        # Empty tuple to force reselection waypoints & airports to be drawn
        self.navsel = ()  # Empty tuple to force reselection waypoints to be drawn
        self.satsel = ()  # Empty tuple to force reselection satellite imagery to be drawn


        # Set up window
        splash.destroy()  # does pg.display.quit()!
        pg.display.init()

        # Full screen
        di = pg.display.Info()
        if self.swfullscreen:
            self.width = di.current_w
            self.height = di.current_h
            reso = (self.width, self.height)
            self.win = pg.display.set_mode(reso, pg.FULLSCREEN)
        else:
            # Windowed
            self.height = int(min(self.height, int(di.current_h * 90 / 100)))
            self.width = int(min(self.width, int(di.current_w * 90 / 100)))
            reso = (self.width, self.height)
            self.win = pg.display.set_mode(reso)

        pg.display.set_caption("BlueSky Open ATM Simulator (F11 = Full Screen)", "BlueSky")
        iconpath = imgpath = "data/graphics/icon.gif"
        iconbmp = pg.image.load(iconpath)
        pg.display.set_icon(iconbmp)

        self.radbmp = self.win.copy()
        self.redrawradbg = True  # Switch to redraw background

        #---------------------RADAR FONTS & EDIT WINDOW-----------------------------
        # Set up fonts
        self.fontrad = Fastfont(self.win, 'Arial', 14, green, False, False)  # name, size, bold,italic
        self.fontamb = Fastfont(self.win, 'Arial', 14, amber, False, False)  # name, size, bold,italic
        self.fontnav = Fastfont(self.win, 'Arial', 12, lightgreyblue, False, False)  # name, size, bold,italic
        self.fontsys = Fastfont(self.win, 'Helvetica', 14, white, False, False)  # name, size, bold,italic

        # Edit window: 6 line of 64 chars
        nch = lst[3]  # number of chars per line
        nlin = lst[4]  # number of lines in windows
        winx = lst[5]  # x-coordinate in pixels of left side
        winy = self.height - lst[6]  # y-coordinate in pixels of bottom
        self.editwin = Console(self.win, nch, nlin, winx, winy)

        # Menu button window
        self.menu = Menu(self.win,10,36)


        #-------------------------COASTLINE DATA--------------------------------------
        # Init geo (coastline)  data
        f = open("data/navdata/coastlines.dat", 'r')
        print("Reading coastlines.dat")
        lines = f.readlines()
        f.close()
        records = []
        for line in lines:
            if not (line.strip() == "" or line.strip()[0] == '#'):
                arg = line.split()
                if len(arg) == 3:
                    records.append([arg[0], float(arg[1]), float(arg[2])])

        # print len(records), " records read."
        # Convert pen up/pen down format of coastlines to numpy arrays with lat/lon

        coastlat0 = []
        coastlon0 = []
        coastlat1 = []
        coastlon1 = []
        clat, clon = -1, -1

        for rec in records:
            lat, lon = rec[1], rec[2]
            if rec[0] == 'D':
                coastlat0.append(clat)
                coastlon0.append(clon)
                coastlat1.append(lat)
                coastlon1.append(lon)
            clat, clon = lat, lon

        self.coastlat0 = np.array(coastlat0)
        self.coastlon0 = np.array(coastlon0)
        self.coastlat1 = np.array(coastlat1)
        self.coastlon1 = np.array(coastlon1)

        del coastlat0, coastlon0, coastlat1, coastlon1  # Clear memory

        self.geosel = ()  # Force reselect first time coastlines
        self.firsel = ()  # Force reselect first time FIR lines

        print("    ", len(self.coastlat0), " coastlines added.")

        # Set default coastlines & FIRs on:
        self.swgeo = True
        self.swfir = True
        self.swgrid = False

        # Route drawing for which acid? "" =  None
        self.acidrte = ""
        self.rtewpid = []
        self.rtewplabel = []

        # User defined background objects
        self.objtype    = []
        self.objcolor   = []
        self.objdata    = []
        self.objname    = []

        # Wpt and apt drawing logic memory
        self.wpswbmp = []              # switch indicating whether label bmp is present
        self.wplabel = []              # List to store bitmaps of label
        self.apswbmp = []              # switch indicating whether label bmp is present
        self.aplabel = []              # List to store bitmaps of label

        self.updateNavBuffers()
        return

    def shownd(self, acid):
        if acid:
            self.ndacid = acid
        self.swnavdisp = not self.swnavdisp

    def updateNavBuffers(self):
        self.wpswbmp = len(bs.navdb.wplat) * [False]
        self.wplabel = len(bs.navdb.wplat) * [0]

        self.apswbmp = len(bs.navdb.aptlat) * [False]
        self.aplabel = len(bs.navdb.aptlat) * [0]

    def echo(self, msg='', flags=0):
        if msg:
            msgs = msg.split('\n')
            for m in msgs:
                self.editwin.echo(m)
        return

    def showssd(self, param):
        return False,"SSD visualization only available in QtGL GUI"

    def cmdline(self, text):
        self.editwin.insert(text)

    def update(self):
        """Draw a new frame"""
        # First check for keys & mouse
        self.keyb.update()
        # Navdisp mode: get center:
        if self.swnavdisp:
            i = bs.traf.id2idx(self.ndacid)
            if i >= 0:
                self.ndlat = bs.traf.lat[i]
                self.ndlon = bs.traf.lon[i]
                self.ndcrs = bs.traf.hdg[i]
            else:
                self.swnavdisp = False
        else:
            self.ndcrs = 0.0


        # Radar window
        # --------------Background--------------

        if self.redrawradbg or self.swnavdisp:
            if self.swnavdisp or not self.swsat:
                self.radbmp.fill(darkgrey)

            else:
                #--------------Satellite image--------------
                navsel = (self.lat0, self.lat1, \
                          self.lon0, self.lon1)
                if not self.satsel == navsel:
                    # Map cutting and scaling: normal case
                    if self.lon1 > self.lon0:
                        x0 = max(0, self.lon0 * self.mapax + self.mapbx)
                        x1 = min(self.mapbmp.get_width() - 1, \
                                 self.lon1 * self.mapax + self.mapbx)

                        y1 = min(self.mapbmp.get_height() - 1, \
                                 self.lat0 * self.mapay + self.mapby)
                        y0 = max(0, self.lat1 * self.mapay + self.mapby)

                        selrect = pg.Rect(x0, y0, abs(x1 - x0), abs(y1 - y0))
                        mapsel = self.mapbmp.subsurface(selrect)
                        self.submap = pg.transform.scale(mapsel, \
                                                         (self.width, self.height))

                        self.radbmp.blit(self.submap, (0, 0))

                    else:
                        # Wrap around case: clip two segments
                        w0 = int(self.width * (180. - self.lon0) / \
                                 (180.0 - self.lon0 + self.lon1 + 180.))
                        w1 = int(self.width - w0)

                        # Left part
                        x0 = max(0, self.lon0 * self.mapax + self.mapbx)
                        x1 = self.mapbmp.get_width() - 1

                        y1 = min(self.mapbmp.get_height() - 1, \
                                 self.lat0 * self.mapay + self.mapby)
                        y0 = max(0, self.lat1 * self.mapay + self.mapby)

                        selrect = pg.Rect(x0, y0, abs(x1 - x0), abs(y1 - y0))
                        mapsel = self.mapbmp.subsurface(selrect)
                        self.submap = pg.transform.scale(mapsel, \
                                                         (w0, self.height))
                        self.radbmp.blit(self.submap, (0, 0))

                        # Right half
                        x0 = 0
                        x1 = min(self.mapbmp.get_width() - 1, \
                                 self.lon1 * self.mapax + self.mapbx)

                        selrect = pg.Rect(x0, y0, abs(x1 - x0), abs(y1 - y0))
                        mapsel = self.mapbmp.subsurface(selrect)
                        self.submap = pg.transform.scale(mapsel, \
                                                         (w1, self.height))
                        self.radbmp.blit(self.submap, (w0, 0))
                        self.submap = self.radbmp.copy()

                    self.satsel = navsel

                else:
                    # Map blit only
                    self.radbmp.blit(self.submap, (0, 0))


            if self.swgrid and not self.swnavdisp:
                # ------Draw lat/lon grid------
                ngrad = int(self.lon1 - self.lon0)

                if ngrad >= 10:
                    step = 10
                    i0 = step * int(self.lon0 / step)
                    j0 = step * int(self.lat0 / step)
                else:
                    step = 1
                    i0 = int(self.lon0)
                    j0 = int(self.lon0)

                for i in range(i0, int(self.lon1 + 1.), step):
                    x, y = self.ll2xy(self.ctrlat, i)
                    pg.draw.line(self.radbmp, lightgreygreen, \
                                 (x, 0), (x, self.height))

                for j in range(j0, int(self.lat1 + 1.), step):
                    x, y = self.ll2xy(j, self.ctrlon)
                    pg.draw.line(self.radbmp, lightgreygreen, \
                                 (0, y), (self.width, y))

            #------ Draw coastlines ------
            if self.swgeo:
                # cx,cy = -1,-1
                geosel = (self.lat0, self.lon0, self.lat1, self.lon1)
                if self.geosel != geosel:
                    self.geosel = geosel

                    self.cstsel = np.where(
                        self.onradar(self.coastlat0, self.coastlon0) + \
                        self.onradar(self.coastlat1, self.coastlon1))

                    # print len(self.cstsel[0])," coastlines"
                    self.cx0, self.cy0 = self.ll2xy(self.coastlat0, self.coastlon0)
                    self.cx1, self.cy1 = self.ll2xy(self.coastlat1, self.coastlon1)

                for i in list(self.cstsel[0]):
                    pg.draw.line(self.radbmp, grey, (self.cx0[i], self.cy0[i]), \
                                 (self.cx1[i], self.cy1[i]))

            #------ Draw FIRs ------
            if self.swfir:
                self.firx0, self.firy0 = self.ll2xy(bs.navdb.firlat0, \
                                                    bs.navdb.firlon0)

                self.firx1, self.firy1 = self.ll2xy(bs.navdb.firlat1, \
                                                    bs.navdb.firlon1)

                for i in range(len(self.firx0)):
                    pg.draw.line(self.radbmp, lightcyan,
                                 (self.firx0[i], self.firy0[i]),
                                 (self.firx1[i], self.firy1[i]))

            # -----------------Waypoint & airport symbols-----------------
            # Check whether we need to reselect waypoint set to be drawn

            navsel = (self.lat0, self.lat1, \
                      self.lon0, self.lon1)
            if self.navsel != navsel:
                self.navsel = navsel

                # Make list of indices of waypoints & airports on screen

                self.wpinside = list(np.where(self.onradar(bs.navdb.wplat, \
                                                           bs.navdb.wplon))[0])

                self.wptsel = []
                for i in self.wpinside:
                    if self.wpsw == 3 or \
                            (self.wpsw == 1 and len(bs.navdb.wpid[i]) == 3) or \
                            (self.wpsw == 2 and bs.navdb.wpid[i].isalpha()):
                        self.wptsel.append(i)
                self.wptx, self.wpty = self.ll2xy(bs.navdb.wplat, bs.navdb.wplon)

                self.apinside = list(np.where(self.onradar(bs.navdb.aptlat, \
                                                           bs.navdb.aptlon))[0])

                self.aptsel = []
                for i in self.apinside:
                    if self.apsw == 2 or (self.apsw == 1 and \
                                                      bs.navdb.aptmaxrwy[i] > 1000.):
                        self.aptsel.append(i)
                self.aptx, self.apty = self.ll2xy(bs.navdb.aptlat, bs.navdb.aptlon)


            #------- Draw waypoints -------
            if self.wpsw > 0:
                # print len(self.wptsel)," waypoints"
                if len(self.wptsel) < self.maxnrwp:
                    wptrect = self.wptsymbol.get_rect()
                    for i in self.wptsel:
                        # wptrect.center = self.ll2xy(bs.navdb.wplat[i],  \
                        #     bs.navdb.wplon[i])
                        wptrect.center = self.wptx[i], self.wpty[i]
                        self.radbmp.blit(self.wptsymbol, wptrect)

                        # If waypoint label bitmap does not yet exist, make it
                        if not self.wpswbmp[i]:
                            self.wplabel[i] = pg.Surface((80, 30), 0, self.win)
                            self.fontnav.printat(self.wplabel[i], 0, 0, \
                                                 bs.navdb.wpid[i])
                            self.wpswbmp[i] = True

                        # In any case, blit it
                        xtxt = wptrect.right + 2
                        ytxt = wptrect.top
                        self.radbmp.blit(self.wplabel[i], (xtxt, ytxt), \
                                         None, pg.BLEND_ADD)

                        if not self.wpswbmp[i]:
                            xtxt = wptrect.right + 2
                            ytxt = wptrect.top

                            # self.fontnav.printat(self.radbmp,xtxt,ytxt, \
                            #     bs.navdb.wpid[i])

            #------- Draw airports -------
            if self.apsw > 0:
                # if len(self.aptsel)<800:
                aptrect = self.aptsymbol.get_rect()

                # print len(self.aptsel)," airports"

                for i in self.aptsel:
                    # aptrect.center = self.ll2xy(bs.navdb.aptlat[i],  \
                    #                            bs.navdb.aptlon[i])
                    aptrect.center = self.aptx[i], self.apty[i]
                    self.radbmp.blit(self.aptsymbol, aptrect)

                    # If airport label bitmap does not yet exist, make it
                    if not self.apswbmp[i]:
                        self.aplabel[i] = pg.Surface((50, 30), 0, self.win)
                        self.fontnav.printat(self.aplabel[i], 0, 0, \
                                             bs.navdb.aptid[i])
                        self.apswbmp[i] = True

                    # In either case, blit it
                    xtxt = aptrect.right + 2
                    ytxt = aptrect.top
                    self.radbmp.blit(self.aplabel[i], (xtxt, ytxt), \
                                     None, pg.BLEND_ADD)

                    # self.fontnav.printat(self.radbmp,xtxt,ytxt, \
                    #     bs.navdb.aptid[i])


            #---------- Draw background trails ----------
            if bs.traf.trails.active:
                bs.traf.trails.buffer()  # move all new trails to background

                trlsel = list(np.where(
                    self.onradar(bs.traf.trails.bglat0, bs.traf.trails.bglon0) + \
                    self.onradar(bs.traf.trails.bglat1, bs.traf.trails.bglon1))[0])

                x0, y0 = self.ll2xy(bs.traf.trails.bglat0, bs.traf.trails.bglon0)
                x1, y1 = self.ll2xy(bs.traf.trails.bglat1, bs.traf.trails.bglon1)

                for i in trlsel:
                    pg.draw.aaline(self.radbmp, bs.traf.trails.bgcol[i], \
                                   (x0[i], y0[i]), (x1[i], y1[i]))

            #---------- Draw ADSB Coverage Area
            if self.swAdsbCoverage:
                # These points are based on the positions of the antennas with range = 200km
                adsbCoverageLat = [53.7863,53.5362,52.8604,51.9538,51.2285,50.8249,50.7382,
                                   50.9701,51.6096,52.498,53.4047,53.6402]
                adsbCoverageLon = [4.3757,5.8869,6.9529,7.2913,6.9312,6.251,5.7218,4.2955,
                                   3.2162,2.7701,3.1117,3.4891]

                for i in range(0,len(adsbCoverageLat)):
                    if i == len(adsbCoverageLat)-1:
                        x0, y0 = self.ll2xy(adsbCoverageLat[i],adsbCoverageLon[i])
                        x1, y1 = self.ll2xy(adsbCoverageLat[0],adsbCoverageLon[0])

                    else:
                        x0, y0 = self.ll2xy(adsbCoverageLat[i],adsbCoverageLon[i])
                        x1, y1 = self.ll2xy(adsbCoverageLat[i+1],adsbCoverageLon[i+1])

                    pg.draw.line(self.radbmp, red,(x0, y0), (x1, y1))

            # User defined objects
            for i in range(len(self.objtype)):

                # Draw LINE or POLYGON with objdata = [lat0,lon,lat1,lon1,lat2,lon2,..]
                if self.objtype[i]=='LINE' or self.objtype[i]=="POLY" or self.objtype[i]=="POLYLINE":
                    npoints = int(len(self.objdata[i])/2)
                    print(npoints)
                    x0,y0 = self.ll2xy(self.objdata[i][0],self.objdata[i][1])
                    for j in range(1,npoints):
                        x1,y1 = self.ll2xy(self.objdata[i][j*2],self.objdata[i][j*2+1])
                        pg.draw.line(self.radbmp,self.objcolor[i],(x0, y0), (x1, y1))
                        x0,y0 = x1,y1

                    if self.objtype[i]=="POLY":
                        x1,y1 = self.ll2xy(self.objdata[i][0],self.objdata[i][1])
                        pg.draw.line(self.radbmp,self.objcolor[i],(x0, y0), (x1, y1))

                # Draw bounding box of objdata = [lat0,lon0,lat1,lon1]
                elif self.objtype[i]=='BOX':
                    lat0 = min(self.objdata[i][0],self.objdata[i][2])
                    lon0 = min(self.objdata[i][1],self.objdata[i][3])
                    lat1 = max(self.objdata[i][0],self.objdata[i][2])
                    lon1 = max(self.objdata[i][1],self.objdata[i][3])

                    x0,y0 = self.ll2xy(lat1,lon0)
                    x1,y1 = self.ll2xy(lat0,lon1)
                    pg.draw.rect(self.radbmp,self.objcolor[i],pg.Rect(x0, y0, x1-x0, y1-y0),1)

                # Draw circle with objdata = [latcenter,loncenter,radiusnm]
                elif self.objtype[i]=='CIRCLE':
                    pass
                    xm,ym     = self.ll2xy(self.objdata[i][0],self.objdata[i][1])
                    xtop,ytop = self.ll2xy(self.objdata[i][0]+self.objdata[i][2]/60.,self.objdata[i][1])
                    radius    = int(round(abs(ytop-ym)))
                    pg.draw.circle(self.radbmp, self.objcolor[i], (int(xm),int(ym)), radius, 1)

            # Reset background drawing switch
            self.redrawradbg = False

        ##############################################################################
        #                          END OF BACKGROUND DRAWING                         #
        ##############################################################################

        # Blit background
        self.win.blit(self.radbmp, (0, 0))

        # Decide to redraw radar picture of a/c
        syst = pg.time.get_ticks() * 0.001
        redrawrad = self.redrawradbg or abs(syst - self.radt0) >= self.radardt

        if redrawrad:
            self.radt0 = syst  # Update lats drawing time of radar


            # Select which aircraft are within screen area
            trafsel = np.where((bs.traf.lat > self.lat0) * (bs.traf.lat < self.lat1) * \
                               (bs.traf.lon > self.lon0) * (bs.traf.lon < self.lon1))[0]

            # ------------------- Draw aircraft -------------------
            # Convert lat,lon to x,y

            trafx, trafy = self.ll2xy(bs.traf.lat, bs.traf.lon)
            trafy -= bs.traf.alt*self.isoalt

            if bs.traf.trails.active:
                ltx, lty = self.ll2xy(bs.traf.trails.lastlat, bs.traf.trails.lastlon)

            # Find pixel size of horizontal separation on screen
            pixelrad=self.dtopix_eq(bs.traf.asas.R/2)

            # Loop through all traffic indices which we found on screen
            for i in trafsel:

                # Get index of ac symbol, based on heading and its rect object
                isymb = int(round((bs.traf.hdg[i] - self.ndcrs) / 6.)) % 60
                pos = self.acsymbol[isymb].get_rect()

                # Draw aircraft symbol
                pos.centerx = trafx[i]
                pos.centery = trafy[i]
                dy = int(self.fontrad.linedy * 7 / 6)

                # Draw aircraft altitude line
                if self.isoalt>1e-7:
                    pg.draw.line(self.win,white,(int(trafx[i]),int(trafy[i])),(int(trafx[i]),int(trafy[i]+bs.traf.alt[i]*self.isoalt)))

                # Normal symbol if no conflict else amber
                toosmall=self.lat1-self.lat0>6 #don't draw circles if zoomed out too much

                if not bs.traf.asas.inconf[i]:
                    self.win.blit(self.acsymbol[isymb], pos)
                    if self.swsep and not toosmall:
                        pg.draw.circle(self.win,green,(int(trafx[i]),int(trafy[i])),pixelrad,1)
                else:
                    self.win.blit(self.ambacsymbol[isymb], pos)
                    if self.swsep and not toosmall:
                        pg.draw.circle(self.win,amber,(int(trafx[i]),int(trafy[i])),pixelrad,1)


                # Draw last trail part
                if bs.traf.trails.active:
                    pg.draw.line(self.win, tuple(bs.traf.trails.accolor[i]),
                                 (ltx[i], lty[i]), (trafx[i], trafy[i]))

                # Label text
                label = []
                if self.swlabel > 0:
                    label.append(bs.traf.id[i])  # Line 1 of label: id
                else:
                    label.append(" ")
                if self.swlabel > 1:
                    if bs.traf.alt[i]>bs.traf.translvl:
                        label.append("FL"+str(int(round(bs.traf.alt[i] / (100.*ft)))))  # Line 2 of label: altitude
                    else:
                        label.append(str(int(round(bs.traf.alt[i] / ft))))  # Line 2 of label: altitude
                else:
                    label.append(" ")
                if self.swlabel > 2:
                    cas = bs.traf.cas[i] / kts
                    label.append(str(int(round(cas))))  # line 3 of label: speed
                else:
                    label.append(" ")


                # Check for changes in traffic label text
                if  not (type(bs.traf.label[i])==list) or \
                      not (type(bs.traf.label[i][3])==str) or \
                        not (label[:3] == bs.traf.label[i][:3]):

                    bs.traf.label[i] = []
                    labelbmp = pg.Surface((100, 60), 0, self.win)
                    if not bs.traf.asas.inconf[i]:
                        acfont = self.fontrad
                    else:
                        acfont = self.fontamb

                    acfont.printat(labelbmp, 0, 0, label[0])
                    acfont.printat(labelbmp, 0, dy, label[1])
                    acfont.printat(labelbmp, 0, 2 * dy, label[2])

                    bs.traf.label[i].append(label[0])
                    bs.traf.label[i].append(label[1])
                    bs.traf.label[i].append(label[2])
                    bs.traf.label[i].append(labelbmp)

                # Blit label
                dest = bs.traf.label[i][3].get_rect()
                dest.top = trafy[i] - 5
                dest.left = trafx[i] + 15
                self.win.blit(bs.traf.label[i][3], dest, None, pg.BLEND_ADD)

                # Draw aircraft speed vectors
                if self.swspd:
                    # just a nominal length: a speed of 150 kts will be displayed
                    # as an arrow of 30 pixels long on screen
                    nomlength    = 30
                    nomspeed     = 150.

                    vectorlength = float(nomlength)*bs.traf.tas[i]/nomspeed

                    spdvcx = trafx[i] + np.sin(np.radians(bs.traf.trk[i])) * vectorlength
                    spdvcy = trafy[i] - np.cos(np.radians(bs.traf.trk[i])) * vectorlength \
                                - bs.traf.vs[i]/nomspeed*nomlength*self.isoalt /   \
                                            self.dtopix_eq(1e5)*1e5

                    pg.draw.line(self.win,green,(trafx[i],trafy[i]),(spdvcx,spdvcy))

            # ---- End of per aircraft i loop


            # Draw conflicts: line from a/c to closest point of approach
            nconf = len(bs.traf.asas.confpairs_unique)
            n2conf = len(bs.traf.asas.confpairs)

            if nconf>0:

                for j in range(n2conf):
                    i = bs.traf.id2idx(bs.traf.asas.confpairs[j][0])
                    if i>=0 and i<bs.traf.ntraf and (i in trafsel):
                        latcpa, loncpa = geo.kwikpos(bs.traf.lat[i], bs.traf.lon[i], \
                                                    bs.traf.trk[i], bs.traf.asas.tcpa[j] * bs.traf.gs[i] / nm)
                        altcpa = bs.traf.lat[i] + bs.traf.vs[i]*bs.traf.asas.tcpa[j]
                        xc, yc = self.ll2xy(latcpa,loncpa)
                        yc = yc - altcpa * self.isoalt
                        pg.draw.line(self.win,amber,(xc,yc),(trafx[i],trafy[i]))

            # Draw selected route:
            if self.acidrte != "":
                i = bs.traf.id2idx(self.acidrte)
                if i >= 0:
                    for j in range(0,bs.traf.ap.route[i].nwp):
                        if j==0:
                            x1,y1 = self.ll2xy(bs.traf.ap.route[i].wplat[j], \
                                               bs.traf.ap.route[i].wplon[j])
                        else:
                            x0,y0 = x1,y1
                            x1,y1 = self.ll2xy(bs.traf.ap.route[i].wplat[j], \
                                               bs.traf.ap.route[i].wplon[j])
                            pg.draw.line(self.win, magenta,(x0,y0),(x1,y1))

                        if j>=len(self.rtewpid) or not self.rtewpid[j]== bs.traf.ap.route[i].wpname[j]:
                            # Waypoint name labels
                            # If waypoint label bitmap does not yet exist, make it

                            # Waypoint name and constraint(s), if there are any
                            txt = bs.traf.ap.route[i].wpname[j]

                            alt = bs.traf.ap.route[i].wpalt[j]
                            spd = bs.traf.ap.route[i].wpspd[j]

                            if alt>=0. or spd >=0.:
                                # Altitude
                                if alt < 0:
                                    txt = txt + " -----/"

                                elif alt > 4500 * ft:
                                    FL = int(round((alt/(100.*ft))))
                                    txt = txt+" FL"+str(FL)+"/"

                                else:
                                    txt = txt+" "+str(int(round(alt / ft))) + "/"

                                # Speed
                                if spd < 0:
                                    txt = txt+"---"
                                else:
                                    txt = txt+str(int(round(spd / kts)))

                            wplabel = pg.Surface((128, 32), 0, self.win)
                            self.fontnav.printat(wplabel, 0, 0, \
                                                 txt)

                            if j>=len(self.rtewpid):
                                self.rtewpid.append(txt)
                                self.rtewplabel.append(wplabel)
                            else:
                                self.rtewpid[j]    = txt
                                self.rtewplabel[j] = wplabel

                        # In any case, blit the waypoint name
                        xtxt = x1 + 7
                        ytxt = y1 - 3
                        self.radbmp.blit(self.rtewplabel[j], (xtxt, ytxt), \
                                         None, pg.BLEND_ADD)

                        # Line from aircraft to active waypoint
                        if bs.traf.ap.route[i].iactwp == j:
                            x0,y0 = self.ll2xy(bs.traf.lat[i],bs.traf.lon[i])
                            pg.draw.line(self.win, magenta,(x0,y0),(x1,y1))



            # Draw aircraft trails which are on screen
            if bs.traf.trails.active:
                trlsel = list(np.where(
                    self.onradar(bs.traf.trails.lat0, bs.traf.trails.lon0) + \
                    self.onradar(bs.traf.trails.lat1, bs.traf.trails.lon1))[0])

                x0, y0 = self.ll2xy(bs.traf.trails.lat0, bs.traf.trails.lon0)
                x1, y1 = self.ll2xy(bs.traf.trails.lat1, bs.traf.trails.lon1)

                for i in trlsel:
                    pg.draw.line(self.win, bs.traf.trails.col[i], \
                                 (x0[i], y0[i]), (x1[i], y1[i]))

                # Redraw background => buffer ; if >1500 foreground linepieces on screen
                if len(trlsel) > 1500:
                    self.redrawradbg = True

        # Draw edit window
        self.editwin.update()

        if self.redrawradbg or redrawrad or self.editwin.redraw:
            self.win.blit(self.menu.bmps[self.menu.ipage], \
                           (self.menu.x, self.menu.y))
            self.win.blit(self.editwin.bmp, (self.editwin.winx, self.editwin.winy))

            # Draw frames
            pg.draw.rect(self.win, white, self.editwin.rect, 1)
            pg.draw.rect(self.win, white, pg.Rect(1, 1, self.width - 1, self.height - 1), 1)

            # Add debug line
            self.fontsys.printat(self.win, 10, 2, str(bs.sim.utc.replace(microsecond=0)))
            self.fontsys.printat(self.win, 10, 18, tim2txt(bs.sim.simt))
            self.fontsys.printat(self.win, 10+80, 2, \
                                 "ntraf = " + str(bs.traf.ntraf))
            self.fontsys.printat(self.win, 10+160, 2, \
                                 "Freq=" + str(int(len(bs.sim.dts) / max(0.001, sum(bs.sim.dts)))))

            self.fontsys.printat(self.win, 10+240, 2, \
                                 "#LOS      = " + str(len(bs.traf.asas.lospairs_unique)))
            self.fontsys.printat(self.win, 10+240, 18, \
                                 "Total LOS = " + str(len(bs.traf.asas.lospairs_all)))
            self.fontsys.printat(self.win, 10+240, 34, \
                                 "#Con      = " + str(len(bs.traf.asas.confpairs_unique)))
            self.fontsys.printat(self.win, 10+240, 50, \
                                 "Total Con = " + str(len(bs.traf.asas.confpairs_all)))

            # Frame ready, flip to screen
            pg.display.flip()

            # If needed, take a screenshot
            if self.screenshot:
                pg.image.save(self.win,self.screenshotname)
                self.screenshot=False
        return


    def ll2xy(self, lat, lon):
        if not self.swnavdisp:
            # RADAR mode:
            # Convert lat/lon to pixel x,y

            # Normal case
            if self.lon1 > self.lon0:
                x = self.width * (lon - self.lon0) / (self.lon1 - self.lon0)

            # Wrap around:
            else:
                dellon = 180. - self.lon0 + self.lon1 + 180.
                xlon = lon + (lon < 0.) * 360.
                x = (xlon - self.lon0) / dellon * self.width

            y = self.height * (self.lat1 - lat) / (self.lat1 - self.lat0)
        else:
            # NAVDISP mode:
            qdr, dist = geo.qdrdist(self.ndlat, self.ndlon, lat, lon)
            alpha = np.radians(qdr - self.ndcrs)
            base = 30. * (self.lat1 - self.lat0)
            x = dist * np.sin(alpha) / base * self.height + self.width / 2
            y = -dist * np.cos(alpha) / base * self.height + self.height / 2

        return np.rint(x), np.rint(y)


    def xy2ll(self, x, y):
        lat = self.lat0 + (self.lat1 - self.lat0) * (self.height - y) / self.height

        if self.lon1 > self.lon0:
            # Normal case
            lon = self.lon0 + (self.lon1 - self.lon0) * x / self.width

        else:
            # Wrap around:
            dellon = 360. + self.lon1 - self.lon0
            xlon = self.lon0 + x / self.width
            lon = xlon - 360. * (lon > 180.)

        # Convert pixel x,y to lat/lan [deg]
        return lat, lon


    def onradar(self, lat, lon):
        """Return boolean (also numpy array) with on or off radar screen (range check)"""

        # Radar mode
        if not self.swnavdisp:
            # Normal case
            if self.lon1 > self.lon0:
                sw = (lat > self.lat0) * (lat < self.lat1) * \
                     (lon > self.lon0) * (lon < self.lon1) == 1

            # Wrap around:
            else:
                sw = (lat > self.lat0) * (lat < self.lat1) * \
                     ((lon > self.lon0) + (lon < self.lon1)) == 1
        # Else NAVDISP mode
        else:
            base = 30. * (self.lat1 - self.lat0)
            dist = geo.latlondist(self.ndlat, self.ndlon, lat, lon) / nm
            sw = dist < base

        return sw



    def zoom(self, factor, absolute = False):
        """Zoom function"""
        oldvalues = self.lat0, self.lat1, self.lon0, self.lon1


        # Zoom factor: 2.0 means halving the display size in degrees lat/lon
        # ZOom out with e.g. 0.5

        ctrlat = (self.lat0 + self.lat1) / 2.
        if not absolute:
             dellat2 = 0.5*(self.lat1 - self.lat0) / factor
        else:
             dellat2 = 1.0/factor

        self.lat0 = ctrlat - dellat2
        self.lat1 = ctrlat + dellat2

        # Normal case
        if self.lon1 > self.lon0:
            ctrlon = (self.lon0 + self.lon1) / 2.
            dellon2 = (self.lon1 - self.lon0) / 2. / factor

        # Wrap around
        else:
            ctrlon = (self.lon0 + self.lon1 + 360.) / 2
            dellon2 = (360. + self.lon1 - self.lon0) / 2. / factor

        if absolute:
            dellon2 = dellat2 * self.width /    \
                (self.height * cos(radians(ctrlat)))

        # Wrap around
        self.lon0 = (ctrlon - dellon2 + 180.) % 360. - 180.
        self.lon1 = (ctrlon + dellon2 + 180.) % 360. - 180.

        # Avoid getting out of range
        if self.lat0 < -90 or self.lat1 > 90.:
            self.lat0, self.lat1, self.lon0, self.lon1 = oldvalues

        self.redrawradbg = True
        self.navsel = ()
        self.satsel = ()
        self.geosel = ()

        return

    def pan(self, *args):
        """Pan function:
               absolute: lat,lon;
               relative: ABOVE/DOWN/LEFT/RIGHT"""
        lat, lon = self.ctrlat, self.ctrlon
        if type(args[0])==str:
            if args[0].upper() == "LEFT":
                lon = lon - 0.5 * (self.lon1 - self.lon0)
            elif args[0].upper() == "RIGHT":
                lon = lon + 0.5 * (self.lon1 - self.lon0)
            elif args[0].upper() == "ABOVE" or args[0].upper() == "UP":
                lat = lat + 0.5 * (self.lat1 - self.lat0)
            elif args[0].upper() == "DOWN":
                lat = lat - 0.5 * (self.lat1 - self.lat0)
            else:
                i = bs.navdb.getwpidx(args[0],self.ctrlat,self.ctrlon)
                if i<0:
                    i = bs.navdb.getaptidx(args[0],self.ctrlat,self.ctrlon)
                    if i>0:
                        lat = bs.navdb.aptlat[i]
                        lon = bs.navdb.aptlon[i]
                else:
                    lat = bs.navdb.wplat[i]
                    lon = bs.navdb.wplon[i]

                if i<0:
                    return False,args[0]+"not found."

        else:
            if len(args)>1:
                lat, lon = args[:2]
            else:
                return False

        # Maintain size & avoid getting out of range
        dellat2 = (self.lat1 - self.lat0) * 0.5
        self.ctrlat = max(min(lat, 90. - dellat2), dellat2 - 90.)

        # Allow wrap around of longitude
        dellon2 = dellat2 * self.width /   \
                                    (self.height * cos(radians(self.ctrlat)))
        self.ctrlon = (lon + 180.) % 360 - 180.

        # Update edge coordinates
        self.lat0 = self.ctrlat - dellat2
        self.lat1 = self.ctrlat + dellat2
        self.lon0 = (self.ctrlon - dellon2 + 180.) % 360. - 180.
        self.lon1 = (self.ctrlon + dellon2 + 180.) % 360. - 180.

        # Redraw background
        self.redrawradbg = True
        self.navsel = ()
        self.satsel = ()
        self.geosel = ()

        # print "Pan lat,lon:",lat,lon
        # print "Latitude  range:",int(self.lat0),int(self.ctrlat),int(self.lat1)
        # print "Longitude range:",int(self.lon0),int(self.ctrlon),int(self.lon1)
        # print "dellon2 =",dellon2

        return True


    def fullscreen(self, switch):  # full screen switch
        """Switch to (True) /from (False) full screen mode"""

        # Reset screen
        pg.display.quit()
        pg.display.init()

        di = pg.display.Info()

        pg.display.set_caption("BlueSky Open ATM Simulator (F11 = Full Screen)",
                               "BlueSky")
        iconpath = imgpath = "data/graphics/icon.gif"
        iconbmp = pg.image.load(iconpath)
        pg.display.set_icon(iconbmp)

        if switch:
            # Full screen mode
            self.width = di.current_w
            self.height = di.current_h
            reso = (self.width, self.height)
            self.win = pg.display.set_mode(reso, pg.FULLSCREEN | pg.HWSURFACE)
        else:
            # Windowed
            self.height = min(self.height, int(di.current_h * 90 / 100))
            self.width = min(self.width, int(di.current_w * 90 / 100))
            reso = (self.width, self.height)
            self.win = pg.display.set_mode(reso)

        # Adjust scaling
        dellat = self.lat1 - self.lat0
        avelat = (self.lat0 + self.lat1) / 2.

        dellon = dellat * self.width / (self.height * cos(radians(avelat)))

        self.lon1 = self.lon0 + dellon

        self.radbmp = self.win.copy()

        # Force redraw and reselect
        self.redrawradbg = True
        self.satsel = ()
        self.navsel = ()
        self.geosel = ()

        return

    def savescreen(self):
        """Save a screenshoot"""
        now=datetime.datetime.now()
        date="%s-%s-%s" % (now.year, now.month, now.day)
        time="time=%sh %sm %ss" % (now.hour, now.minute, now.second)
        num=1
        while self.session == "new":
            self.folder="./screenshots/"+date+"-session-"+str(num)
            if os.path.exists(self.folder):
                num+=1
            else:
                os.makedirs(self.folder)
                self.session=num
        self.screenshotname=self.folder+"/"+time+".bmp"
        self.screenshot=True


    def ltopix_eq(self,lat):
        """
        Latitude to pixel conversion. Compute how much pixels a
        degree in latlon is in longitudinal direction.
        """

        pwidth=self.width
        lwidth=self.lon1-self.lon0

        return int(lat/lwidth*pwidth)


    def dtopix_eq(self,dist):
        """
        Distance to pixel conversion. Compute how much pixels a
        meter is in longitudinal direction
        """
        lat=dist/111319.
        return self.ltopix_eq(lat)

    def color(self, name, r, g, b):
        ''' Set custom color for aircraft or shape. '''
        if areafilter.hasArea(name):
            idx = self.objname.index(name)
            self.objcolor[idx] = (r, g, b)
        else:
            return False, 'No object found with name ' + name

        self.redrawradbg = True  # redraw background
        return True

    def objappend(self,itype,name,data):
        """Add user defined objects"""
        if data is None:
            return self.objdel()


        self.objname.append(name)
        self.objtype.append(itype)
        if self.objtype[-1]==1:
            self.objtype[-1]="LINE" # Convert to string

        self.objcolor.append(cyan)
        self.objdata.append(data)


        self.redrawradbg = True  # redraw background

        return

    def objdel(self):
        """Add user defined objects"""
        self.objname     = []
        self.objtype     = []
        self.objcolor    = []
        self.objdata     = []
        self.redrawradbg = True  # redraw background
        return

    def showroute(self, acid):  # Toggle show route for an aircraft id
        if self.acidrte == acid:
            self.acidrte = ""  # Click twice on same: route disappear
        else:
            self.acidrte = acid  # Show this route
        return True


    def addnavwpt(self,name,lat,lon): # Draw new navdb waypoint
        # As in pygame navdb has already updated data, simply redraw background
        self.wpswbmp.append(False) # Add cell to buffer
        self.wplabel.append(0) # Add cell to buffer
        self.redrawradbg = True  # redraw background
        return

    def getviewctr(self):
        return (self.ctrlat, self.ctrlon)

    def getviewbounds(self): # Return current viewing area in lat, lon
        return self.lat0, self.lat1, self.lon0, self.lon1

    def drawradbg(self): # redraw radar background
        self.redrawradbg = True
        return

    def filteralt(self, *args):
        return False, 'Filteralt not implemented in Pygame gui'

    def feature(self,sw,arg=""):
        # Switch/toggle/cycle radar screen features e.g. from SWRAD command
        # Coastlines
        if sw == "GEO":
            self.swgeo = not self.swgeo


        # FIR boundaries
        elif sw == "FIR":
            self.swfir = not self.swfir

        # Airport: 0 = None, 1 = Large, 2= All
        elif sw == "APT":
            self.apsw = (self.apsw + 1) % 3
            if not (arg== ""):
                self.apsw = int(cmdargs[2])
            self.navsel = []

        # Waypoint: 0 = None, 1 = VOR, 2 = also WPT, 3 = Also terminal area wpts
        elif sw == "VOR" or sw == "WPT" or sw == "WP" or sw == "NAV":
            self.wpsw = (self.wpsw + 1) % 4
            if not (arg== ""):
                self.wpsw = int(arg)
            self.navsel = []

        # Satellite image background on/off
        elif sw == "SAT":
            self.swsat = not self.swsat

        elif sw[:4] == "ADSB":
            self.swAdsbCoverage = not self.swAdsbCoverage

        # Traffic labels: cycle nr of lines 0,1,2,3
        elif sw[:3] == "LAB":  # Nr lines in label
            self.swlabel = (self.swlabel + 1) % 4
            if not (arg == ""):
                self.swlabel = int(arg)

        else:
            self.redrawradbg = False
            return False # switch not found

        self.redrawradbg = True
        return True # Success

    def show_file_dialog(self):
        return opendialog()
    def symbol(self):
        self.swsep = not self.swsep
        return True

    def show_cmd_doc(self, cmd=''):
        # Show documentation on command
        if not cmd:
            cmd = 'Command-Reference'
        curdir = os.getcwd()
        os.chdir("data/html")
        htmlfile = cmd.lower()+".html"
        if os.path.isfile(htmlfile):
            try:
                subprocess.Popen(htmlfile,shell=True)
            except:
                os.chdir(curdir)
                return False,"Opening "+htmlfile+" failed."
        else:
            os.chdir(curdir)
            return False,htmlfile+" is not yet available, try HELP PDF or check the wiki on Github."

        os.chdir(curdir)
        return True,"HTML window opened"
