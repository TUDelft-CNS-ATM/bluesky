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
from aero import cas2tas, mach2tas, kts


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


def txt2tim(txt):
    """Convert text to time in seconds:
       HH
       HH:MM
       HH:MM:SS
       HH:MM:SS.hh
    """
    timlst = txt.split(":")

    t = 0.

    # HH
    if len(timlst[0])>0 and timlst[0].isdigit():
        t = t+3600.*int(timlst[0])
        
    # MM
    if len(timlst)>1 and len(timlst[1])>0 and timlst[1].isdigit():
        t = t+60.*int(timlst[1])
    
    # SS.hh
    if len(timlst)>2 and len(timlst[2])>0:
        if timlst[2].replace(".","0").isdigit():
            t = t + float(timlst[2])
    
    return t
    
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


def col2rgb(txt):
    cols = {"black": (0, 0, 0), "white": (255, 255, 255), "green": (0, 255, 0),
            "red": (255, 0, 0), "blue": (0, 0, 255), "magenta": (255, 0, 255),
            "yellow": (240, 255, 127), "amber": (255, 255, 0), "cyan": (0, 255, 255)}
    try:
        rgb = cols[txt.lower().strip()]
    except:
        rgb = cols["white"]  # default

    return rgb


def degto180(angle):
    """Change to domain -180,180 """
    return (angle + 180.) % 360 - 180.

def degtopi(angle):
    """Change to domain -pi,pi """
    return (angle + pi) % (2.*pi) - pi


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


def cmdsplit(cmdline, trafids=None):
    cmdline = cmdline.strip()
    if len(cmdline) == 0:
        return '', []

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

    # If a traffic id list is passed, check if command and first argument need to be switched
    if trafids and len(cmdargs) > 1 and trafids.count(cmdargs[0]):
        cmdargs[0:2] = cmdargs[1::-1]

    # return command, argumentlist
    return cmdargs[0], cmdargs[1:]


def txt2lat(lattxt):
    """txt2lat: input txt: N52'14'13.5 or N52"""
    txt = lattxt.upper().replace("N", "").replace("S", "-")  # North positive, South negative
    neg = txt.count("-") > 0
    if txt.count("'") > 0 or txt.count('"') > 0:
        txt = txt.replace('"', "'")  # replace " by '
        degs = txt.split("'")
        div = 1
        lat = 0
        if neg:
            f = -1.
        else:
            f = 1.
        for xtxt in degs:
            if len(xtxt) > 0:
                try:
                    lat = lat + f * abs(float(xtxt)) / float(div)
                    div = div * 60
                except:
                    print "txt2lat value error:",lattxt
                    return 0.0                    
    else:
        lat = float(txt)
    return lat
    # Return float


def txt2lon(lontxt):
    """txt2lat: input txt: N52'14'13.5 or N52"""
    # It should first be checked if lontxt is a regular float, to avoid removing
    # the 'e' in a scientific-notation number.
    try: 
        lon = float(lontxt)

    # Leading E will trigger error ansd means simply East,just as  W = West = Negative
    except:
        
        txt = lontxt.upper().replace("E", "").replace("W", "-")  # East positive, West negative
        neg = txt.count("-") > 0

        # Use of "'" and '"' as degrees/minutes/seconds
        # Also "N52'"
        if txt.count("'") > 0 or txt.count('"') > 0:
            txt = txt.replace('"', "'")  # replace " by '
            degs = txt.split("'")
            div = 1
            lon = 0.0
            if neg:
                f = -1.
            else:
                f = 1.
            for xtxt in degs:
                if len(xtxt)>0.0:
                    try:
                       lon = lon + f * abs(float(xtxt)) / float(div)
                    except:
                       print "txt2lon value error:",lontxt
                       return 0.0 

                div = div * 60
        else:  # Cope with "W65"without "'" or '"', also "-65" or "--65" 
            try:
                neg = txt.count("-") > 0
                if neg:
                    f = -1.
                else:
                    f = 1.
                lon = f*abs(float(txt))
            except:
                print "txt2lon value error:",lontxt
                return 0.0 

    return lon


def deg180(dangle):
    """ Convert any difference in angles to interval [ -180,180 ) """
    return (dangle + 180.) % 360. - 180.
