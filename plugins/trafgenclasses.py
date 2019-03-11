import random
from math import degrees,radians,cos,sin,atan2,sqrt


from bluesky import stack,traf,sim,tools,navdb
from bluesky.tools.position import txt2pos
from bluesky.tools.geo import kwikqdrdist,kwikpos,kwikdist,latlondist,qdrdist
from bluesky.tools.misc import degto180,txt2alt,txt2spd
from bluesky.tools.aero import nm,ft

# Default values
swcircle = False
ctrlat = 52.6   # [deg]
ctrlon = 5.4    # [deg]
radius = 230.   # [nm]

def setcircle(ictrlat,ictrlon,iradius):
    global ctrlat,ctrlon,radius,swcircle
    swcircle = True
    ctrlat, ctrlon, radius = ictrlat,ictrlon,iradius
    return


class Source():
    # Define a source
    def __init__(self,name,cmd,cmdargs):
        self.name    = name.upper()
        self.tupdate = sim.simt-999.
        self.flow    = 0
        self.incircle = True
        self.segdir  = None  # Segment direction in degrees
        self.hdg     = None # When runway is used as separate airport

        # Is location a circle segment?
        if swcircle and self.name[:4]=="SEGM":
            self.type = "seg"
            self.lat,self.lon,brg = getseg(self.name)
            pass

        else:
            success, posobj = txt2pos(name,ctrlat,ctrlon)
            if success:
                # for runway type, get heading as default optional argument for command line
                if posobj.type == "rwy":
                    aptname, rwyname = name.split('/RW')
                    rwyname = rwyname.lstrip('Y')
                    try:
                        self.hdg = navdb.rwythresholds[aptname][rwyname][2]
                    except:
                        self.hdg = None
                        pass
                else:
                    rwyhdg = None


                self.lat,self.lon = posobj.lat, posobj.lon
                self.type = posobj.type

                # If outside circle change lat,lon to edge of circle
                self.incircle = incircle(self.lat,self.lon)
                if not self.incircle:
                    self.type = "seg"
                    self.segdir,dist = qdrdist(ctrlat,ctrlon,posobj.lat,posobj.lon)
                    segname = "SEGM"+str(int(round(self.segdir)))
                    self.lat,self.lon,hdg = getseg(segname)

            else:
                print("ERROR: trafgen class Source called for "+name+". Position not found")
                self.lat,self.lon = 0.0,0.0


        # Aircraft types
        self.actypes = ["*"]

        # Runways
        self.runways    = [] # name
        self.rwylat     = []
        self.rwylon     = []
        self.rwyhdg     = []
        self.rwyline    = [] # number of aircraft waiting in line
        self.rwytotime  = [] # time of last takeoff

        # If source type is a runway, add it as the only runway
        if self.type=="rwy":
            self.runways = [rwyname]
            self.rwylat     = [self.lat]
            self.rwylon     = [self.lon]
            self.rwyhdg     = [int(rwyname.rstrip("LCR").lstrip("0"))*10.]
            self.rwyline    = [] # number of aircraft waiting in line
            self.rwytotime  = [-999] # time of last takeoff


        self.dtakeoff = 90. # sec take-off interval on one runway, default 90 sec

        # Destinations
        self.dest        = []
        self.destlat     = []
        self.destlon     = []
        self.desttype    = [] # "apt","wpt","seg"
        self.desthdg     = []  # if dest is a runway (for flights within FIR) or circle segment (source outside FIR)
        self.destactypes = []   # Types for this destinations ([]=use defaults for this source)

        #Names of drawing objects runways
        self.polys       = []    # Names of current runway polygons to remove when runways change

        # Limits on start values alt,spd,hdg
        self.startaltmin = None
        self.startaltmax = None
        self.startspdmin = None
        self.startspdmax = None
        self.starthdgmin = None
        self.starthdgmax = None

        return

    def setrunways(self,cmdargs):
        self.runways   = []
        self.rwylat    = []
        self.rwylon    = []
        self.rwyhdg    = []
        self.rwyline   = []
        self.rwytotime = []

        for runwayname in cmdargs:
            if runwayname[0] == "R":
                success, rwyposobj = txt2pos(self.name + "/" + runwayname, self.lat, self.lon)
            else:
                success,rwyposobj = txt2pos(self.name+"/RW"+runwayname,self.lat,self.lon)
            if success:
                self.runways.append(runwayname)
                self.rwylat.append(rwyposobj.lat)
                self.rwylon.append(rwyposobj.lon)
                rwyname = runwayname.upper().lstrip('RWY')
                #try:
                if True:
                    self.rwyhdg.append(navdb.rwythresholds[self.name][rwyname][2])
                    #if self.name=="EHAM":
                    #S    print("runway added with hdg:",self.rwyhdg[-1])
                #except:
                #    success = False
                self.rwyline.append(0)
                self.rwytotime.append(-999.)
            else:
                self.runways.append(runwayname)
                self.rwylat.append(self.lat)
                self.rwylon.append(self.lon)
                rwydigits = runwayname.lstrip("RWY").rstrip("LCR")
                self.rwyhdg.append(10. * int(rwydigits.rstrip("LCR").lstrip("0")))
                self.rwyline.append(0)
                self.rwytotime.append(-999.)

    def setalt(self,cmdargs):
        if len(cmdargs)==1:
            alt = txt2alt(cmdargs[0])
            self.startaltmin = alt
            self.startaltmax = alt
        elif len(cmdargs)>1:
            alt0,alt1 = txt2alt(cmdargs[0]),txt2alt(cmdargs[1])
            self.startaltmin = min(alt0,alt1)
            self.startaltmax = max(alt0,alt1)
        else:
            stack.stack("ECHO "+self.name+" ALT "+str(self.startaltmin)+" "+str(self.startaltmax))

    def setspd(self,cmdargs):
        if len(cmdargs)==1:
            spd = txt2spd(cmdargs[0])
            self.startspdmin = spd
            self.startapdmax = spd
        elif len(cmdargs)>1:
            spd0,spd1 = txt2spd(cmdargs[0]),txt2spd(cmdargs[1])
            self.startspdmin = min(spd0,spd1)
            self.startspdmax = max(spd0,spd1)
        else:
            stack.stack("ECHO "+self.name+" SPD "+str(self.startaltmin)+" "+str(self.startaltmax))

    def sethdg(self,cmdargs):
        if len(cmdargs)==1:
            hdg = float(cmdargs[0])
            self.starthdgmin = hdg
            self.starthdgmax = hdg
        elif len(cmdargs)>1:
            hdg0,hdg1 = float(cmdargs[0]),float(cmdargs[1])
            hdg0,hdg1 = min(hdg0,hdg1),max(hdg0,hdg1)
            if hdg1-hdg0>180.:
                hdg0,hdg1 = hdg1-360,hdg0
            self.starthdgmin = hdg0
            self.starthdgmax = hdg1
        else:
            stack.stack("ECHO ERROR "+self.name+" HDG "+str(self.starthdgmin)+" "+str(self.starthdgmax))

    def adddest(self,cmdargs):
        # Add destination with a given aircraft types
        destname = cmdargs[0]
        try:
            freq = int(cmdargs[1])
        except:
            freq = 1

        # Get a/c types frequency list, if given
        destactypes = []
        if len(cmdargs) > 2:  # also types are given for this destination
            for c in cmdargs[2:]:
                if c.count(":") > 0:
                    actype, typefreq = c.split(":")
                else:
                    actype = c
                    typefreq = "1"

                for f in range(int(typefreq)):
                    destactypes.append(actype)

        if not destname[:4]=="SEGM":
            success,posobj = txt2pos(destname,self.lat,self.lon)
            if success:
                if posobj.type == "rwy":
                    for i in range(freq):
                        self.dest.append(destname)
                        self.destlat.append(posobj.lat)
                        self.destlon.append(posobj.lon)
                        self.desthdg.append(None)
                        self.desttype.append(posobj.type)
                        self.destactypes.append(destactypes)

                else:
                    for i in range(freq):
                        aptname = destname
                        self.dest.append(destname)
                        self.destlat.append(posobj.lat)
                        self.destlon.append(posobj.lon)
                        self.desthdg.append(0)
                        self.desttype.append(posobj.type)
                        self.destactypes.append(destactypes)
            else:
                # Add random destination
                for i in range(freq):
                    name = "SEGM" + str(int(random.random() * 360.))
                    lat, lon, hdg = getseg(name)
                    self.dest.append(name)
                    self.destlat.append(lat)
                    self.destlon.append(lon)
                    self.desthdg.append(hdg)
                    self.desttype.append("seg")
                    self.destactypes.append(destactypes)

        else:
            for i in range(freq):
                # Segment as destination, bearing from center = heading
                lat,lon,hdg = getseg(destname)
                self.dest.append(destname)
                self.destlat.append(lat)
                self.destlon.append(lon)
                self.desthdg.append(hdg)
                self.desttype.append("seg")
                self.destactypes.append(destactypes)

        return True

    def setflow(self,flowtxt):
        try:
            self.flow = float(flowtxt) #in a/c per hour, also starts flow as it by default zero
        except:
            return False
        return True


    def addactypes(self,actypelist):
        self.actypes = self.actypes + makefreqlist(actypelist)

    def update(self,gain):
        # Time step update of source

        # Get time step
        dt = sim.simt - self.tupdate
        self.tupdate = sim.simt

        # Time for a new aircraft?
        if dt>0.0:

            # Calculate probability of a geberate occurring with flow
            chances = 1.0-gain*self.flow*dt/3600. #flow is in a/c per hour=360 seconds
            if random.random() >= chances:

                # Runways defined? => use runway lines (queues)
                if len(self.runways)>0:
                    # We do not yet need to create an aircraft
                    gennow = False

                    # Find shortest line and put it in
                    isel = random.randint(0,len(self.runways)-1)
                    self.rwyline[isel] = self.rwyline[isel] + 1
                else:
                    # Yes we do need to generate one now
                    gennow = True
            else:
                gennow = False

            # Check for generating aircraft
            # First check runways for a/c already in line:
            txt = ""
            for i in range(len(self.runways)):
                # Runway vacated and there is one waiting?
                if sim.simt-self.rwytotime[i]>self.dtakeoff and self.rwyline[i]>0:
                    #if self.name == "EHAM":
                    #   print(sim.simt, self.runways[i], self.rwytotime[i])
                    self.rwytotime[i] = sim.simt
                    self.rwyline[i]   = self.rwyline[i]-1

                    # Choose and aicraft type, check for distance
                    if len(self.dest)>0:
                        idest = int(random.random() * len(self.dest))
                    else:
                        idest = -1

                    if idest>=0:
                        acid = randacname(self.name,self.dest[idest])

                        if self.desttype[idest]=="seg" or self.dest[idest][:4]=="SEGM":
                            lat,lon,hdg = getseg(self.dest[idest])
                        else:
                            success,posobj = txt2pos(self.dest[idest],ctrlat,ctrlon)
                            lat,lon = posobj.lat,posobj.lon
                        distroute = latlondist(self.lat,self.lon,lat,lon)/nm

                    else:
                        acid = randacname(self.name, self.name)

                    if self.destactypes[idest] == []:
                        actype = random.choice(self.actypes)
                        actype = checkactype(actype, distroute, self.actypes)
                    else:
                        actype = random.choice(self.destactypes[idest])

                    stack.stack("CRE "+",".join([acid, actype,
                                                 str(self.rwylat[i]),str(self.rwylon[i]),str(self.rwyhdg[i]),
                                                 "0.0","0.0"]))

                    #wplat,wplon = kwikpos(self.rwylat[i],self.rwylon[i],self.rwyhdg[i],5.0*nm)
                    #stack.stack(acid + " ADDWPT ",wplat," ",wplon)
                    #stack.stack(acid+"LNAV ON")
                    if idest>=0:
                        if self.dest[idest][:4] != "SEGM":
                            stack.stack(acid + " DEST " + self.dest[idest])
                        else:
                            stack.stack(acid + " DEST " + str(self.destlat[idest])
                                    + " " + str(self.destlon[idest]))

                    if self.name[:4] != "SEGM":
                        stack.stack(acid + " ORIG " + self.name)
                    else:
                        stack.stack(acid + " ORIG " + str(self.lat) + " " + str(self.lon))

                    stack.stack(acid + " SPD 250")
                    stack.stack(acid + " ALT FL100")
                    stack.stack(acid + " HDG " + str(self.rwyhdg[i]))

                    stack.stack(acid+" LNAV OFF")

            # Not runway, then define instantly at position with random heading or in case of segment inward heading
            if gennow:
                if not self.incircle:
                    lat,lon = kwikpos(ctrlat,ctrlon,self.segdir,radius)
                    hdg = self.segdir-180
                elif self.type=="seg":
                    lat,lon,brg = getseg(self.name)
                    hdg = (brg+180)%360
                elif self.type=="rwy":
                    lat,lon = self.lat,self.lon
                    hdg     = self.hdg # Runway heading
                else:
                    hdg = random.random()*360.

                if self.startaltmin and self.startaltmax:
                    alt = random.randint(int(self.startaltmin), int(self.startaltmax))
                else:
                    alt = random.randint(200, 300) * 100 * ft

                if self.startspdmin and self.startspdmax:
                    spd = random.randint(int(self.startspdmin), int(self.startspdmax))
                else:
                    spd = random.randint(250, 350)

                alttxt, spdtxt = "FL" + str(int(round(alt / (100 * ft)))), str(spd)

                # Add destination
                if len(self.dest)>0:
                    idest = int(random.random() * len(self.dest))
                    acid = randacname(self.name,self.dest[idest])
                else:
                    acid  = randacname(self.name,self.name)
                    idest = -1

                stack.stack("CRE " + ",".join([acid, random.choice(self.actypes),
                                               str(self.lat), str(self.lon), str(int(hdg%360)),
                                               alttxt,spdtxt]))

                if idest>=0:
                    if self.dest[idest][:4] != "SEGM":
                        stack.stack(acid + " DEST " + self.dest[idest])
                    else:
                        stack.stack(acid + " DEST " + str(self.destlat[idest])+" "+str(self.destlon[idest]))

                if self.name[:4] != "SEGM":
                    stack.stack(acid + " ORIG " + self.name)
                else:
                    stack.stack(acid + " ORIG " + str(self.lat)+" "+str(self.lon))

                if alttxt=="0" and spdtxt =="0":
                    stack.stack(acid+" SPD 250")
                    stack.stack(acid+" ALT FL100")
                else:
                    if idest>=0:
                        if self.desttype[idest] == "seg":
                            lat, lon, hdg = getseg(self.dest[idest])
                            brg, dist = kwikdist(self.lat, self.lon, lat, lon)
                            stack.stack(acid + " HDG " + str(brg))
                        else:
                            stack.stack(acid + " LNAV ON")
                            #stack.stack(acid + " VNAV ON")

class Drain():
    # Define a drain: destination within area, source outside
    def __init__(self,name,cmd,cmdargs):
        self.name    = name.upper()
        self.tupdate = sim.simt-999.
        self.flow    = 0
        self.incircle = True
        self.segdir  = None  # Segment direction in degrees

        # Is location a circle segment?
        if self.name[:4]=="SEGM":
            self.type = "seg"
            self.lat,self.lon,brg = getseg(self.name) # For SEGMnnn to SEGMnnn for crossing flights optional
            pass

        else:
            success, posobj = txt2pos(name,ctrlat,ctrlon)
            if success:
                # for runway type, get heading as default optional argument for command line
                if posobj.type == "rwy":
                    aptname, rwyname = name.split('/RW')
                    rwyname = rwyname.lstrip('Y')
                    try:
                        rwyhdg = navdb.rwythresholds[aptname][rwyname][2]
                    except:
                        rwyhdg = None
                        pass
                else:
                    rwyhdg = None


                self.lat,self.lon = posobj.lat, posobj.lon
                self.type = posobj.type

                # If outside circle change lat,lon to edge of circle
                self.incircle = incircle(self.lat,self.lon)
                if not self.incircle:
                    self.type = "seg"
                    self.segdir,dist = qdrdist(ctrlat,ctrlon,posobj.lat,posobj.lon)
                    segname = "SEGM"+str(int(round(self.segdir)))
                    self.lat,self.lon,hdg = getseg(segname)

            else:
                print("ERROR: trafgen class Drain called for "+name+". Position not found")
                self.lat,self.lon = 0.0,0.0


        # Aircraft types
        self.actypes = ["*"]

        # Runways
        self.runways    = [] # name
        self.rwylat     = []
        self.rwylon     = []
        self.rwyhdg     = []
        self.rwyline    = [] # number of aircraft waiting in line

        #Origins
        self.orig        = []
        self.origlat     = []
        self.origlon     = []
        self.origtype    = [] # "apt","wpt","seg"
        self.orighdg     = []  # if orig is a runway (for flights within FIR) or circle segment (drain outside FIR)
        self.origactypes = []   # Types for this originations ([]=use defaults for this drain)
        self.origincirc  = []

        # Limits on start values alt,spd,hdg
        self.startaltmin = None
        self.startaltmax = None
        self.startspdmin = None
        self.startspdmax = None
        self.starthdgmin = None
        self.starthdgmax = None

        #Names of drawing objects runways
        self.polys       = []

        return

    def setrunways(self,cmdargs):
        self.runways = []
        self.rwylat  = []
        self.rwylon  = []
        self.rwyhdg  = []
        for runwayname in cmdargs:
            if runwayname[0] == "R":
                success, rwyposobj = txt2pos(self.name + "/" + runwayname, self.lat, self.lon)
            else:
                success,rwyposobj = txt2pos(self.name+"/RW"+runwayname,self.lat,self.lon)
            if success:
                self.runways.append(runwayname)
                self.rwylat.append(rwyposobj.lat)
                self.rwylon.append(rwyposobj.lon)
                rwyname = runwayname.upper().lstrip('RWY').lstrip("RW")
                try:
                    self.rwyhdg.append(navdb.rwythresholds[self.name][rwyname][2])
                except:
                    success = False

    def addorig(self,cmdargs):
        # Add origin with a given aircraft types
        origname = cmdargs[0]
        try:
            freq = int(cmdargs[1])
        except:
            freq = 1

        # Get a/c types frequency list, if given
        origactypes = []
        if len(cmdargs) > 2:  # also types are given for this origin
            for c in cmdargs[2:]:
                if c.count(":") > 0:
                    actype, typefreq = c.split(":")
                else:
                    actype = c
                    typefreq = "1"

                for f in range(int(typefreq)):
                    origactypes.append(actype)

        if not origname[:4]=="SEGM":
            success,posobj = txt2pos(origname,self.lat,self.lon)
            if success:
                incirc = incircle(posobj.lat, posobj.lon)
                if posobj.type == "rwy":
                    for i in range(freq):
                        self.orig.append(origname)
                        self.origlat.append(posobj.lat)
                        self.origlon.append(posobj.lon)
                        self.orighdg.append(None)
                        self.origtype.append(posobj.type)
                        self.origactypes.append(origactypes)
                        self.origincirc.append(incirc)

                else:
                    for i in range(freq):
                        self.orig.append(origname)
                        self.origlat.append(posobj.lat)
                        self.origlon.append(posobj.lon)
                        self.orighdg.append(None)
                        self.origtype.append(posobj.type)
                        self.origactypes.append(origactypes)
                        self.origincirc.append(incirc)
            else:
                for i in range(freq):
                    name = "SEGM" + str(int(random.random() * 360.))
                    lat, lon, hdg = getseg(name)
                    self.orig.append(name)
                    self.origlat.append(lat)
                    self.origlon.append(lon)
                    self.orighdg.append(None)
                    self.origtype.append("seg")
                    self.origactypes.append(origactypes)
                    self.origincirc.append(False)
        else:
            # Segment as origin, bearing from center = heading
            lat,lon,hdg = getseg(origname)
            self.orig.append(origname)
            self.origlat.append(lat)
            self.origlon.append(lon)
            self.orighdg.append(hdg)
            self.origtype.append("seg")
            self.origactypes.append(origactypes)
            self.origincirc.append(incircle(lat,lon))
        return True

    def setalt(self,cmdargs):
        if len(cmdargs)==1:
            alt = txt2alt(cmdargs[0])
            self.startaltmin = alt
            self.startaltmax = alt
        elif len(cmdargs)>1:
            alt0,alt1 = txt2alt(cmdargs[0]),txt2alt(cmdargs[1])
            self.startaltmin = min(alt0,alt1)
            self.startaltmax = max(alt0,alt1)
        else:
            stack.stack("ECHO "+self.name+" ALT "+str(self.startaltmin)+" "+str(self.startaltmax))

    def setspd(self,cmdargs):
        if len(cmdargs)==1:
            spd = txt2spd(cmdargs[0])
            self.startspdmin = spd
            self.startapdmax = spd
        elif len(cmdargs)>1:
            spd0,spd1 = txt2spd(cmdargs[0]),txt2spd(cmdargs[1])
            self.startspdmin = min(spd0,spd1)
            self.startspdmax = max(spd0,spd1)
        else:
            stack.stack("ECHO "+self.name+" SPD "+str(self.startaltmin)+" "+str(self.startaltmax))

    def sethdg(self,cmdargs):
        if len(cmdargs)==1:
            hdg = float(cmdargs[0])
            self.starthdgmin = hdg
            self.starthdgmax = hdg
        elif len(cmdargs)>1:
            hdg0,hdg1 = float(cmdargs[0]),float(cmdargs[1])
            hdg0,hdg1 = min(hdg0,hdg1),max(hdg0,hdg1)
            if hdg1-hdg0>180.:
                hdg0,hdg1 = hdg1-360,hdg0
            self.starthdgmin = hdg0
            self.starthdgmax = hdg1
        else:
            stack.stack("ECHO "+self.name+" HDG "+str(self.starthdgmin)+" "+str(self.starthdgmax))



    def setflow(self,flowtxt):
        try:
            self.flow = float(flowtxt) #in a/c per hour, also starts flow as it by default zero
        except:
            return False
        return True

    def addactypes(self,actypelist):
        self.actypes = self.actypes + makefreqlist(actypelist)

    def update(self,gain):
        # Time step update of drain

        # Get time step
        dt = sim.simt - self.tupdate
        self.tupdate = sim.simt

        # Time for a new aircraft?
        if dt>0.0:

            # Calculate probability of a geberate occurring with flow
            chances = 1.0-gain*self.flow*dt/3600. #flow is in a/c per hour=360 seconds
            if random.random() >= chances:
                # Calculate starting position using origin
                if len(self.orig)>0:
                    # Add origin
                    iorig = int(random.random() * len(self.orig))
                else:
                    iorig = -1

                if iorig>=0:
                    incirc = self.origincirc[iorig]
                    lat,lon = self.origlat[iorig],self.origlon[iorig]
                    hdg,dist = qdrdist(lat,lon,self.lat,self.lon)
                else:
                    print("Warning update drain",self.name,"called with no origins present!")
                    hdg = random.random()*360.
                    print("using random segment",int(hdg+180)%360)

                    incirc = False

                if not incirc:
                    lat,lon = kwikpos(ctrlat,ctrlon,(hdg+180)%360,radius)
                elif self.origtype=="seg":
                    lat,lon,brg = getseg(self.name)
                    hdg = (brg+180)%360
                else:
                    hdg = random.random()*360.

                if incirc and (self.origtype[iorig]=="apt" or self.origtype[iorig]=="rwy"):
                    alttxt,spdtxt = str(0),str(0)
                else:
                    if self.startaltmin and self.startaltmax:
                        alt = random.randint(int(self.startaltmin), int(self.startaltmax))
                    else:
                        alt = random.randint(200,300)*100*ft

                    if self.startspdmin and self.startspdmax:
                        spd = random.randint(int(self.startspdmin), int(self.startspdmax))
                    else:
                        spd = random.randint(250,350)

                    alttxt,spdtxt = "FL"+str(int(round(alt/(100*ft)))), str(spd)

                if iorig>=0:
                    acid = randacname(self.orig[iorig], self.name)
                else:
                    acid = randacname("LFPG", self.name)

                if len(self.origactypes)>0:
                    actype = random.choice(self.origactypes[iorig])
                else:
                    actype = random.choice(self.actypes)

                stack.stack("CRE " + ",".join([acid,actype,str(lat), str(lon),
                                               str(int(hdg%360)),alttxt,spdtxt]))
                if iorig>=0:
                    if self.orig[iorig][:4]!="SEGM":
                        stack.stack(acid + " ORIG " + self.orig[iorig])
                    else:
                        stack.stack(acid + " ORIG " + str(self.origlat[iorig]) + " " +\
                                     str(self.origlon[iorig]))
                if not (self.name[:4]=="SEGM"):
                    stack.stack(acid + " DEST " + self.name)
                else:
                    stack.stack(acid + " ADDWPT " + str(self.lat) + " " + str(self.lon))

                if alttxt=="0" and spdtxt =="0":
                    stack.stack(acid+" SPD 250")
                    stack.stack(acid+" ALT FL100")
                    #stack.stack(acid+" LNAV ON")
                else:
                    stack.stack(acid + " LNAV ON")
                    #stack.stack(acid + " VNAV ON") ATC discretion

def randacname(orig,dest):
    companies = 70*["KLM"]+30*["TRA"]+10*["**"]+["PH"]
    company = random.choice(companies)
    if dest[:2]=="EH" and orig[:2]=="EH":
        company = "PH"

    if company=="**":
        company = chr(ord("A") + int(random.random()*26))+ \
                  chr(ord("A") + int(random.random()*26))

    #if orig=="EHKD"

    # Make flight number or Dutch call sign for VFR traffic
    firstx =  True
    while firstx or (acname in traf.id):
        if not (company=="PH"):
            fltnr = str(int(random.random()*900+100))
        else:
            fltnr = "-" + chr(ord("A") + int(random.random()*26))+ \
                chr(ord("A") + int(random.random() * 26)) + \
                chr(ord("A") + int(random.random() * 26))

        acname = company+fltnr
        firstx = False

    return company+fltnr

def makefreqlist(txtlist):
    # Expand freqlist
    # Translate  arguments ["KL:3","HV:1","PH","MP:5"] into a pick list with the relative frequency:
    #           ["KL","KL","KL","HV","PH","MP","MP","MP","MP","MP"]
    lst = []
    for item in txtlist:
        if item.count(":"):
            itemtxt,freq = item.strip().split(":")
            for i in range(int(freq)):
                lst.append(itemtxt)
        else:
            lst.append(item)
    return lst


def getseg(txt):

    # Get a random position on the segment with the heading inward
    # SEGMnnn is segment in direction nnn degrees from center circle

    brg = float(txt[4:])

    lat,lon = kwikpos(ctrlat,ctrlon,brg,radius)

    return lat,lon,brg


def checkactype(curtype,dist,alltypes):
    # Use lookup table/size/weight to match flight distance with a suitable aircraft type (if available)
    # To avoid B744s flying to Eelde and PA28 to New York
    newtype = curtype

    # Temporary quick fix
    # TBD
    if curtype[0]=="A" or curtype[0]=="B": # Likely Airbus or Boeing? => no small distance < 300 nm
        if dist<300:
            # Try to find another one maxiumum 25 times
            n = 25
            while n>0 and newtype[0] in ["A","B"]:
                n = n - 1
                newtype = random.choice(alltypes)
    else:
        if dist > 500:
            # Try to find another one maxiumum 25 times
            n = 25
            while n > 0 and not( newtype[0] in ["A", "B"]):
                n = n - 1
                newtype = random.choice(alltypes)

    return newtype

def incircle(lat,lon): # Check whether position is inside current circle definition
    if not swcircle:
        return True
    else:
        dist = latlondist(ctrlat,ctrlon,lat,lon)/nm
        return dist<=radius


def crosscircle(xm,ym,r,xa,ya,xb,yb):
    # Find crossing points of line throught a end b and circle around m with radius r
    a1 = xb-xa
    a2 = yb-ya
    b1 = xb-xm
    b2 = yb-ym
    A = a1*a1+a2*a2
    B = 2*a1*b1+2*a2*b2
    C = b1*b1+b2*b2-r*r
    D = B*B-4*A*C
    if D<0:
        return [None,None]
    else:
        VD = sqrt(D)
        lam1 = (-B-VD)/(2*A)
        x1,y1 = b1+lam1*a1+xm , b2+lam1*a2+ym
        lam2 = (-B+VD)/(2*A)
        x2, y2 = b1+lam2*a1+xm , b2+lam2*a2+ym
        return [[x1,y1],[x2,y2]]
