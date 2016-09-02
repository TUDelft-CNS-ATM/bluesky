# -*- coding: utf-8 -*-

from misc import txt2lat, txt2lon

def txt2pos(txt,traf,navdb,reflat,reflon):
    """ Convert text with lat/lon, waypoint,airport,runway,navaid,fix etc. 
    to a position with lat,lon,name and type info
    """

    # Join texts incase of tuple/list
    if type(txt)==tuple or type(txt)==list:
        name = ",".join(txt).upper()
    else:
        name = txt.upper()

    # Check for two args or one
    if name.count(",")>0:
        nargs = 2
    else:
        nargs = 1

    return Position(name,traf,navdb,reflat,reflon),nargs

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
    def __init__(self,name,traf,navdb,reflat,reflon):
 
        if name.count(",")>0: #lat,lon or apt,rwy type
            txt1,txt2 = name.split(",")
            if islat(txt1):
                self.lat = txt2lat(txt1)
                self.lon = txt2lon(txt2)
                self.name = ""
                self.type ="latlon"

            # apt-RWY type
            else:
                idx = self.navdb.apid.index(txt.upper())
 
                self.lat = navdb.aplat[idx]
                self.lon = navdb.aplon[idx]
                # What should this return? for now, return airport
                # What is this?  
                # arglist = traf.navdb.rwythresholds[args[argidx]][rwyname][:2]
                self.name = self.navdb.apid[idx]
#                self.type = "rwy"
                self.type ="apt" # temporarily let stack handle runway part

        else:
            self.name = name
            idx = traf.id2idx(name)
            if idx>0:
                self.lat = traf.lat[idx]
                self.lon = traf.lon[idx]
                self.name = ""
                self.type = "ac"
    
            else:            
                idx = navdb.getwpidx(name, reflat, reflon)
                if idx >= 0:
                    self.lat  = navdb.wplat[idx]
                    self.lon  = navdb.wplon[idx]
                    self.type = "wpt"
            
                else:
                    idx = navdb.getapidx(name)
                    if idx>0:
                        self.lat = navdb.aplat[idx]
                        self.lon = navdb.aplon[idx]
                        self.type = "apt"
    
    
#        if self.type=="ac" or self.type=="latlon":
#            # Make N52E004 name
#            clat = "SN"[lat>0]
#            clon = "WE"[lon>0] 
#            name = clat + "%02d"%int(abs(round(lat))) + \
#                   clon + "%03d"%int(abs(round(lon)))

        return


