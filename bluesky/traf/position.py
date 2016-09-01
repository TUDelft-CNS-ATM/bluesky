# -*- coding: utf-8 -*-

def txt2pos(txt,navdb):
    """ Convert text with lat/lon, waypoint,airport,runway,navaid,fix etc. 
    to a position with lat,lon,name and type info
    """
    return Position(txt,navdb)

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
    def __init__(self,intxt,navdb):

        # Join texts incase of tuple/list
        if type(txt)=tuple or type(txt)=list:
            txt = ",".join(intxt)
        else:
            txt = intxt

        # TODO: insert parsing logic   

        # Analyze using navdb



    
    
    
        # Fill members
        self.lat     =  0.0  # [deg]
        self.lon     =  0.0 # [deg]
        self.type    = "wpt" # string
        self.subtype = "vor" #string
        self.orgname = txt   #input name
        self.name    = "" # name empty in case of lat/lon or N52E004 type name
        
        return


