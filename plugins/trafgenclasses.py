import random
from math import degrees,radians,cos,sin,atan2,sqrt


from bluesky import stack,traf,sim,tools,navdb
from bluesky.tools.position import txt2pos
from bluesky.tools.geo import kwikqdrdist,kwikpos,kwikdist,latlondist,qdrdist
from bluesky.tools.misc import degto180

# Default values
swcircle = False
ctrlat = 52.6 # [deg]
ctrlon = 5.4  # [deg]
radius = 230. # [nm]

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

        # Is location a circle segment?
        if swcircle and self.name[:4]=="SEGM":
            self.type = "seg"
            self.lat,self.lon,brg = getseg(self.name)# TBD : Do we need this as we also have sorce outside cirle?
            # Yes: segn to segn for crossing flights optional
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
                print("ERROR: contestclass called for "+name+". Position not found")
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

        self.dtakeoff = 90. # sec take-off interval on one runway, default 90 sec

        # Destinations
        self.dest        = []
        self.destlat     = []
        self.destlon     = []
        self.desttype    = [] # "apt","wpt","seg"
        self.desthdg     = []  # if dest is a runway (for flights within FIR) or circle segment (source outside FIR)
        self.destactypes = []   # Types for this destinations ([]=use defaults for this source)



        # When this roucres is created it could be with runways or with destinations
        if cmd == "RUNWAY" or cmd == "RWY":
            self.addrunways(cmdargs)
        elif cmd == "DEST":
            self.adddest(cmdargs)
        elif cmd =="TYPES":
            self.addactypes(cmdargs)
        elif cmd=="FLOW":
            self.setflow(cmdargs[0])
        return

    def addrunways(self,cmdargs):
        for runwayname in cmdargs:
            success,rwyposobj = txt2pos(self.name+"/"+runwayname,self.lat,self.lon)
            if success:
                self.runways.append(runwayname)
                self.rwylat.append(rwyposobj.lat)
                self.rwylon.append(rwyposobj.lon)
                rwyname = runwayname.upper().strip('RWY').strip("RW")
                self.rwyhdg.append(navdb.rwythresholds[self.name][rwyname][2])
                self.rwyline.append(0)
                self.rwytotime.append(-999.)
                # TBD draw runways

    def adddest(self,cmdargs):
        # Add destination with a given aicraft types
        destname = cmdargs[0]
        freq = int(cmdargs[1])

        # Get a/c types frequency list, if given
        destactypes = []
        if len(cmdargs) >= 2:  # also types are given for this destination
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
            # Segment as destination, bearing from center = heading
            lat,lon,hdg = getseg(destname)
            self.dest.append(destname)
            self.destlat.append(lat)
            self.destlon.append(lon)
            self.desthdg.append(hdg)
            self.desttype.append("seg")
            self.destactypes.append(destactypes)

    def setflow(self,flowtxt):
        self.flow = float(float(flowtxt)) #in a/c per hour, also starts flow as it by default zero


    def addactypes(self,actypelist):
        self.actypes = self.actypes + makefreqlist(actypelist)

    def update(self):
        # Time step update of source

        # Get time step
        dt = sim.simt - self.tupdate
        self.tupdate = sim.simt

        # Time for a new aircraft?
        if dt>0.0:

            # Calculate probability of a geberate occurring with flow
            chances = 1.0-self.flow*dt/3600. #flow is in a/c per hour=360 seconds
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

                    # Choose and aicraft type, chck for distance
                    idest = int(random.random() * len(self.dest))

                    acid = randacname(self.name,self.dest[idest])

                    if self.desttype[idest]=="seg":
                        lat,lon,hdg = getseg(self.dest[idest])
                    else:
                        success,posobj = txt2pos(self.dest[idest],ctrlat,ctrlon)
                        lat,lon = posobj.lat,posobj.lon
                    distroute = latlondist(self.lat,self.lon,lat,lon)

                    if self.destactypes[idest]==[]:
                        actype = random.choice(self.actypes)
                        actype = checkactype(actype,distroute,self.actypes)
                    else:
                        actype = random.choice(self.destactypes[idest])

                    stack.stack("CRE "+",".join([acid, actype,
                                                 str(self.rwylat[i]),str(self.rwylon[i]),str(self.rwyhdg[i]),
                                                 "0.0","0.0"]))
                    stack.stack(" ".join([acid,"SPD","250"]))
                    stack.stack(" ".join([acid,"ALT","5000"]))
                    # Add waypoint for after take-off

                    if self.desttype[idest]=="seg":
                        lat,lon,hdg = getseg(self.dest[idest])
                        brg,dist = kwikqdrdist(self.lat,self.lon,lat,lon)
                        #stack.stack(acid+" HDG "+str(brg))
                    else:
                        stack.stack(acid+" DEST "+self.dest[idest])
                        stack.stack(acid+" LNAV OFF")
                        #stack.stack(acid+" VNAV ON")

            # Not runway, then define instantly at position with random heading or in case of segment inward heading
            if gennow:
                if not self.incircle:
                    lat,lon = kwikpos(ctrlat,ctrlon,self.segdir,radius)
                    hdg = self.segdir-180
                elif self.type=="seg":
                    lat,lon,brg = getseg(self.name)
                    hdg = (brg+180)%360
                else:
                    hdg = random.random()*360.

                if (self.type=="apt" or self.type=="rwy") and self.incircle:
                    alt,spd = str(0),str(0)
                else:
                    alt,spd = "FL"+str(random.randint(200,300)), str(random.randint(250,350))
                # Add destination
                idest = int(random.random() * len(self.dest))

                acid = randacname(self.name,self.dest[idest])

                stack.stack("CRE " + ",".join([acid, random.choice(self.actypes),
                                               str(self.lat), str(self.lon), str(int(hdg)),
                                               alt,spd]))

                if alt=="0" and spd =="0":
                    stack.stack(" ".join([acid, "SPD", "250"]))
                    stack.stack(" ".join([acid, "ALT", "5000"]))
                    #stack.stack(acid+" LNAV ON")
                else:
                    if self.desttype[idest] == "seg":
                        lat, lon, hdg = getseg(self.dest[idest])
                        brg, dist = kwikdist(self.lat, self.lon, lat, lon)
                        stack.stack(acid + " HDG " + str(brg))
                    else:
                        stack.stack(acid + " DEST " + self.dest[idest])
                        stack.stack(acid + " LNAV ON")
                        #stack.stack(acid + " VNAV ON")

def randacname(orig,dest):
    companies = 70*["KL"]+30*["HV"]+10*["**"]+["PH"]
    company = random.choice(companies)
    if dest[:2]=="EH" and orig[:2]=="EH":
        company = "PH"

    if company=="**":
        company = chr(ord("A") + int(random.random()*26))+ \
                  chr(ord("A") + int(random.random()*26))

    # Make flight number or Dutch call sign for VFR traffic
    if not (company=="PH"):
        fltnr = str(int(random.random()*900+100))
    else:
        fltnr = "-" + chr(ord("A") + int(random.random()*26))+ \
                chr(ord("A") + int(random.random() * 26)) + \
                chr(ord("A") + int(random.random() * 26))

    acname = company+fltnr
    i = 0
    while (acname in traf.id):
        fltnr = fltnr+str(i)
        i     = i + 1
        acname = company+ fltnr

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

def incircle(lat,lon): # Chekc whether position is inside circle definition
    if not swcircle:
        return True
    else:
        dist = latlondist(ctrlat,ctrlon,lat,lon)
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
