"""
Miscellaneous modules

Modules:
     txt2alt(txt): read altitude[ft] from txt (FL ot ft)
     txt2spd(spd,h): read CAS or Mach and convert to TAS for given altitude
     tim2txt(t)  : convert time[s] to HH:MM:SS.hh
     i2txt(i,n)  : convert integer to string of n chars with leading zeros

Created by  : Jacco M. Hoekstra
"""

from numpy import *
from time import strftime, gmtime
from aero import cas2tas, mach2tas, kts, nm


def txt2alt(txt):
    """Convert text to altitude in ft: also FL300 => 30000. as float"""
    # First check for FL otherwise feet
    if txt.upper()[:2] == 'FL' and len(txt) >= 4:  # Syntax check Flxxx or Flxx
        try:
            return 100. * int(txt[2:])
        except:
            return -999.
    else:
        try:
            return float(txt)
        except:
            return -999.
    return -999


def tim2txt(t):
    """Convert time to timestring: HH:MM:SS.hh"""
    return strftime("%H:%M:%S.", gmtime(t)) + i2txt(int((t - int(t)) * 100.), 2)


def i2txt(i, n):
    """Convert integer to string with leading zeros to make it n chars long"""
    itxt = str(i)
    return "0" * (n - len(itxt)) + itxt


def txt2spd(txt, h):
    """Convert text to speed (EAS [kts]/MACH[-] to TAS[m/s])"""
    if len(txt) == 0:
        return -1.
    try:
        if txt[0] == 'M':
            M_ = float(txt[1:])
            if M_ >= 20:   # Handle M95 notation as .95
                M_ = M_ * 0.01
            acspd = mach2tas(M_, h)  # m/s

        elif txt[0] == '.' or (len(txt) >= 2 and txt[:2] == '0.'):
            spd_ = float(txt)
            acspd = mach2tas(spd_, h)  # m/s

        else:
            spd_ = float(txt) * kts
            acspd = cas2tas(spd_, h)  # m/s
    except:
        return -1.

    return acspd


def kwikdist(lata, lona, latb, lonb):
    """
    Convert text to altitude: 4500 = 4500 ft,
    Return altitude in meters
    Quick and dirty dist [nm] inreturn
    """

    re = 6371000.  # readius earth [m]
    dlat = array(radians(latb - lata))
    dlon = array(radians(lonb - lona))
    cavelat = array(cos(radians(lata + latb) / 2.))

    dangle = sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist = re * dangle / nm

    return mat(dist)


def kwikqdrdist(lata, lona, latb, lonb):
    """Gives quick and dirty qdr[deg] and dist [nm]
       (note: does not work well close to poles)"""

    re = 6371000.  # radius earth [m]
    dlat = array(radians(latb - lata))
    dlon = array(radians(degto180(lonb - lona)))
    cavelat = array(cos(radians(lata + latb) / 2.))

    dangle = sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist = re * dangle

    qdr = degrees(arctan2(dlon * cavelat, dlat)) % 360.

    return mat(qdr), mat(dist)


def col2rgb(txt):
    cols = {"black": (0, 0, 0),  "white": (255, 255, 255), "green": (0, 255, 0),
            "red": (255, 0, 0),  "blue": (0, 0, 255),      "magenta": (255, 0, 255),
            "yellow": (240, 255, 127), "amber": (255, 255, 0),  "cyan": (0, 255, 255)}
    try:
        rgb = cols[txt.lower().strip()]
    except:
        rgb = cols["white"]  # default

    return rgb


def degto180(angle):
    """Change to domain -180,180 """
    return (angle + 180.) % 360 - 180.


def findnearest(lat, lon, latarr, lonarr):
    """Find index of nearest postion in numpy arrays with lat and lon"""
    if len(latarr) > 0 and len(latarr) == len(lonarr):
        coslat = cos(radians(lat))
        dy = radians(lat - latarr)
        dx = coslat * radians(degto180(lon - lonarr))
        d2 = dx * dx + dy * dy
        idx = list(d2).index(d2.min())

        return idx
    else:
        return -1


def cmdsplit(cmdline):
    # Use both comma and space as a separator: two commas mean an empty argument
    while cmdline.find(",,") >= 0:
        cmdline = cmdline.replace(",,", ",@,")  # Mark empty arguments

    # Replace comma's by space
    cmdline = cmdline.replace(",", " ")

    # Split using spaces
    cmdargs = cmdline.split()  # Make list of cmd arguments

    # Adjust for empty arguments
    for i in range(len(cmdargs)):
        if cmdargs[i] == "@":
            cmdargs[i] = ""

    return cmdargs

def txt2lat(lattxt):
    """txt2lat: input txt: N52'14'13.5 or N52"""
    txt = lattxt.replace("N","").replace("S","-") # North positive, sout negative    
    if txt.count("'")>0 or txt.count('"')>0:
        txt = txt.replace('"',"'") # replace " by '
        degs = txt.split("'")
        div = 1
        lat = 0
        for deg in degs:
            if len(deg)>0:
                lat = lat+float(deg)/float(div)
                div = div*60
    else:
        lat = float(txt)
    return lat
    # Return float
        
def txt2lon(lontxt):
    """txt2lat: input txt: N52'14'13.5 or N52"""
    txt = lontxt.replace("E","").replace("W","-") # East positive, West negative    
    if txt.count("'")>0 or txt.count('"')>0:
        txt = txt.replace('"',"'") # replace " by '
        degs = txt.split("'")
        div = 1
        lon = 0
        for deg in degs:
            if len(deg)>0:
                lon = lon+float(deg)/float(div)
                div = div*60
    else:
        lon = float(txt)
    return lon
