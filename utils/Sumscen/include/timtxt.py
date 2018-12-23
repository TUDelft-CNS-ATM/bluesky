from time import strftime, gmtime

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

