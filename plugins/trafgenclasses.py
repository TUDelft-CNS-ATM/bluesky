import random

from bluesky import stack,traf,sim,tools,navdb
from bluesky.tools.position import txt2pos
from bluesky.tools.geo import kwikqdrdist,kwikpos,latlondist

# Default values
ctrlat = 52.6
ctrlon = 5.4
radius = 230.

def setcircle(ictrlat,ictrlon,iradius):
    global ctrlat,ctrlon,radius
    ctrlat, ctrlon, radius = ictrlat,ictrlon,iradius
    return


class Source():
    # Define a source or drain
    def __init__(self,name,cmd,cmdargs):
        self.name    = name.upper()
        self.tupdate = sim.simt-999.
        self.flow    = 0

        # Is location a circle segment or airport?
        if self.name[:3]=="SEG":
            self.type = "seg"
            # TBD : Do we need this as we also have drains? Yes: segn to segn for crossing flights
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

        self.dtakeoff = 90. # sec take-off interval on one runway

        # Destinations
        self.dest     = []
        self.destlat  = []
        self.destlon  = []
        self.desttype = [] # "apt","wpt","seg"
        self.desthdg  = []  # if dest is a runway (for flights within FIR)



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

    def adddest(self,cmdargs):
        # Add destination with a given frequency
        destname = cmdargs[0]
        freq = int(cmdargs[1])
        if not destname[:3]=="SEG":
            success,posobj = txt2pos(destname,self.lat,self.lon)
            if success:
                if posobj.type == "rwy":
                    for i in range(freq):
                        self.dest.append(destname)
                        self.destlat.append(posobj.lat)
                        self3.destlon.append(posobj.lon)
                        self.desthdg.append(None)
                        self.desttype.append(posobj.type)

                else:
                    for i in range(freq):
                        aptname = destname
                        self.dest.append(destname)
                        self.destlat.append(posobj.lat)
                        self.destlon.append(posobj.lon)
                        self.desthdg.append(0)
                        self.desttype.append(posobj.type)
        else:
            lat,lon,hdg = getseg(destname)
            self.dest.append(destname)
            self.destlat.append(lat)
            self.destlon.append(lon)
            self.desthdg.append(hdg)
            self.desttype.append("seg")





    def setflow(self,flowtxt):
        self.flow = float(float(flowtxt)) #in a/c per hour, also starts flow as it by default zero


    def addactypes(self,actypelist):
        self.actypes = self.actypes + makefreqlist(actypelist)

    def update(self):
        # Time stwp update of source


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
            # First check runways:
            txt = ""
            for i in range(len(self.runways)):
                # Runway vacated and there is one waiting?
                if sim.simt-self.rwytotime[i]>self.dtakeoff and self.rwyline[i]>0:
                    self.rwytotime[i] = sim.simt
                    self.rwyline[i]=self.rwyline[i]-1
                    acid = randacname()

                    # Choose and aicraft type, chck for distance
                    actype    = random.choice(self.actypes)
                    idest = int(random.random() * len(self.dest))
                    if self.desttype[idest]=="seg":
                        lat,lon,hdg = getseg(self.dest[idest])
                    else:
                        success,posobj = txt2pos(self.dest[idest],ctrlat,ctrlon)
                        lat,lon = posobj.lat,posobj.lon
                    distroute = latlondist(self.lat,self.lon,lat,lon)
                    actype = checkactype(actype,distroute,self.actypes)

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
                if self.type=="seg":
                    self.lat,self.lon,hdg = getseg(self.name)
                else:
                    hdg = random.random()*360.
                acid = randacname()
                if self.type=="apt" or "rwy":
                    alt,spd = str(0),str(0)
                else:
                    alt,spd = "FL300", "350"

                stack.stack("CRE " + ",".join([acid, random.choice(self.actypes),
                                               str(self.lat), str(self.lon), str(int(hdg)),
                                               alt,spd]))
                # Add destination
                idest = int(random.random() * len(self.dest))
                if alt=="0" and spd =="0":
                    stack.stack(" ".join([acid, "SPD", "250"]))
                    stack.stack(" ".join([acid, "ALT", "5000"]))
                    #stack.stack(acid+" LNAV ON")
                if self.desttype[idest] == "seg":
                    lat, lon, hdg = getseg(self.dest[idest])
                    brg, dist = kwikdist(self.lat, self.lon, lat, lon)
                    stack.stack(acid + " HDG " + str(brg))
                else:
                    stack.stack(acid + " DEST " + self.dest[idest])
                    stack.stack(acid + " LNAV OFF")
                    #stack.stack(acid + " VNAV ON")



class Drain(): # TBD
    def __init__(self,name,cmd,cmdargs):
        pass

    def update(self):
        pass


def randacname():
    companies = 70*["KL"]+30*["HV"]+10*["**"]+["PH"]
    company = random.choice(companies)

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
    # Translate ["KL:3","HV:1","PH","MP:5"] into:
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
    iseg = int(txt[3:])

    brg = (iseg-1)*30.+random.random()*30.
    hdg = (brg+180)%360.

    lat,lon = kwikpos(ctrlat,ctrlon,brg,radius)

    return lat,lon,hdg


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



