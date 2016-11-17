# -*- coding: utf-8 -*-

from misc import txt2lat, txt2lon

def txt2pos(name,traf,navdb,reflat,reflon):
    pos = Position(name.upper().strip(),traf,navdb,reflat,reflon)
    if not pos.error:
        return True,pos
    else:
        return False,name+" not found in database"

def islat(txt):
    # Is it a latitude-like format or not?

    # Take out non-digit chars which are allowed
    testtxt = txt.upper().strip().strip("-")   \
                         .strip("+").strip("\n").strip(",") \
                         .replace('"',"").replace("'","").replace(".","")

    # Take away one leading N or S if present before other chars
    if (testtxt[0]=="N" or testtxt[0]=="S") and len(testtxt)>1:
        testtxt =testtxt[1:]

    return testtxt.isdigit()

class Position():
    """ Position class: container for position data
    """

    # position types: "latlon","nav","apt","rwy"    
    
    # Initialize using text
    def __init__(self,name,traf,navdb,reflat,reflon):

        self.name = name # default: copy source name
        self.error = False # we're optmistic about our succes

        # lat,lon type ?
        if name.count(",")>0: #lat,lon or apt,rwy type
            txt1,txt2 = name.split(",")
            if islat(txt1):
                self.lat = txt2lat(txt1)
                self.lon = txt2lon(txt2)
                self.name = ""
                self.type ="latlon"

        # runway type ? "EHAM/RW06","EHGG/RWY27"
        elif name.count("/RW")>0:
            aptname,rwytxt = name.split("/")

            rwyname = rwytxt.strip("RW").strip("Y").strip().upper() # remove RW or RWY and spaces

            self.lat,self.lon = traf.navdb.rwythresholds[aptname][rwyname][:2] # raises error if not found
            self.type = "rwy" 

        # airport?
        elif navdb.aptid.count(name)>0:
            idx = navdb.aptid.index(name.upper())

            self.lat = navdb.aptlat[idx]
            self.lon = navdb.aptlon[idx]
            self.type ="apt"

        # fix or navaid?
        elif navdb.wpid.count(name)>0:
            idx = navdb.getwpidx(name,reflat,reflon)
            self.lat = navdb.wplat[idx]
            self.lon = navdb.wplon[idx]
            self.type ="nav"

        # aircraft id?
        elif traf.id2idx(name)>=0:
            idx = traf.id2idx(name)
            self.name = ""
            self.type = "latlon"
            self.lat = traf.lat[idx]
            self.lon = traf.lon[idx]
            
            # exception for pan, check for LEFT, RIGHT, ABOVE or DOWN
        elif name.upper() in ["LEFT","RIGHT","ABOVE","DOWN"]:
            self.lat = reflat
            self.lon = reflon
            self.type = "dir"

# Not used now, but save this code for future use
#            # Make a N52E004 type waypoint name
#            clat = "SN"[lat>0]
#            clon = "WE"[lon>0] 
#            name = clat + "%02d"%int(abs(round(lat))) + \
#                   clon + "%03d"%int(abs(round(lon)))
        else:
            self.error = True            
            # raise error with missing data... (empty position object)

        return


