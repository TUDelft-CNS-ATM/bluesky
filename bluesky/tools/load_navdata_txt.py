from ..settings import data_path
from aero import ft
import numpy as np
import os


def load_navdata_txt():
    #---------- Read waypoints.dat file ----------
    wptdata         = dict()
    wptdata['wpid']    = []              # identifier (string)
    wptdata['wplat']   = []              # latitude [deg]
    wptdata['wplon']   = []              # longitude [deg]
    wptdata['wpapt']   = []              # reference airport {string}
    wptdata['wptype']  = []              # type (string)
    wptdata['wpco']    = []              # two char country code (string)
    with open(data_path + "/global/waypoints.dat", "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example line:
            # ABARA, , 61.1833, 50.85, UUYY, High and Low Level, RS
            #  [id]    [lat]    [lon]  [airport]  [type] [country code]
            #   0  1     2       3         4        5         6
            fields = line.split(",")
            wptdata['wpid'].append(fields[0].strip())  # id, no leading or trailing spaces

            wptdata['wplat'].append(float(fields[2]))  # latitude [deg]
            wptdata['wplon'].append(float(fields[3]))  # longitude [deg]

            wptdata['wpapt'].append(fields[4].strip())  # id, no leading or trailing spaces
            wptdata['wptype'].append(fields[5].strip().lower())    # type
            wptdata['wpco'].append(fields[6].strip())     # country code

    wptdata['wplat']   = np.array(wptdata['wplat'])
    wptdata['wplon']   = np.array(wptdata['wplon'])

    #----------  Read airports.dat file ----------
    aptdata           = dict()
    aptdata['apid']      = []              # 4 char identifier (string)
    aptdata['apname']    = []              # full name
    aptdata['aplat']     = []              # latitude [deg]
    aptdata['aplon']     = []              # longitude [deg]
    aptdata['apmaxrwy']  = []              # reference airport {string}
    aptdata['aptype']    = []              # type (int, 1=large, 2=medium, 3=small)
    aptdata['apco']      = []              # two char country code (string)
    with open(data_path + "/global/airports.dat", "r") as f:
        types = {'L': 1, 'M': 2, 'S': 3}
        for line in f:
            line = line.strip()
            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example line:
            # EHAM, SCHIPHOL, 52.309, 4.764, Large, 12467, NL
            #  [id]   [name] [lat]    [lon]  [type] [max rwy length in ft] [country code]
            #   0        1     2        3       4          5                   6
            fields = line.split(",")

            # Skip airports without identifier in file and closed airports
            if fields[0].strip() == "" or fields[4].strip() == 'Closed':
                continue

            aptdata['apid'].append(fields[0].strip())  # id, no leading or trailing spaces
            aptdata['apname'].append(fields[1].strip())  # name, no leading or trailing spaces

            aptdata['aplat'].append(float(fields[2]))  # latitude [deg]
            aptdata['aplon'].append(float(fields[3]))  # longitude [deg]

            aptdata['aptype'].append(types[fields[4].strip()[0]])  # large=1, medium=2, small=3

            # Not all airports have rwy length (e.g. heliports)
            try:
                aptdata['apmaxrwy'].append(float(fields[5])*ft)  # max rwy ltgh [m]
            except:
                aptdata['apmaxrwy'].append(0.0)

            aptdata['apco'].append(fields[6].strip().lower()[:2])     # country code

    aptdata['aplat']    = np.array(aptdata['aplat'])
    aptdata['aplon']    = np.array(aptdata['aplon'])
    aptdata['apmaxrwy'] = np.array(aptdata['apmaxrwy'])
    aptdata['aptype']   = np.array(aptdata['aptype'])

    #----------  Read FIR files ----------
    firdata         = dict()
    firdata['fir']     = []
    firdata['firlat0'] = []
    firdata['firlon0'] = []
    firdata['firlat1'] = []
    firdata['firlon1'] = []

    files = os.listdir(data_path + "/global/fir")

    # Get fir names
    for filname in files:
        if ".txt" in filname:
            firname = filname[:filname.index(".txt")]
            firdata['fir'].append([firname, [], []])

            with open(data_path + "/global/fir/" + filname, "r") as f:
                for line in f:
                    rec = line.upper().strip()

                    if len(rec) == 0:
                        continue

                    latsign = 2 * int(line[0] == "N") - 1
                    latdeg  = float(line[1:4])
                    latmin  = float(line[5:7])
                    latsec  = float(line[8:14])
                    lat     = latsign*latdeg+latmin/60.+latsec/3600.

                    lonsign = 2 * int(line[15] == "E") - 1
                    londeg  = float(line[16:19])
                    lonmin  = float(line[20:22])
                    lonsec  = float(line[23:29])
                    lon     = lonsign*londeg+lonmin/60.+lonsec/3600.

                    # For drawing create a line from last lat,lon to current lat,lon
                    if len(firdata['fir'][-1][1]) > 0:  # skip first lat,lon
                        firdata['firlat0'].append(firdata['fir'][-1][1][-1])
                        firdata['firlon0'].append(firdata['fir'][-1][2][-1])
                        firdata['firlat1'].append(lat)
                        firdata['firlon1'].append(lon)

                    # Add to FIR record
                    firdata['fir'][-1][1].append(lat)
                    firdata['fir'][-1][2].append(lon)

    # Convert lat/lon lines to numpy arrays
    firdata['firlat0'] = np.array(firdata['firlat0'])
    firdata['firlat1'] = np.array(firdata['firlat1'])
    firdata['firlon0'] = np.array(firdata['firlon0'])
    firdata['firlon1'] = np.array(firdata['firlon1'])

    return wptdata, aptdata, firdata
