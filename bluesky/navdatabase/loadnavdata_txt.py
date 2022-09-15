''' Load navigation data from text files.'''
import numpy as np
from zipfile import ZipFile
from bluesky import settings
import bluesky as bs
from bluesky.tools.aero import ft


REARTH_INV = 1.56961231e-7


def loadnavdata_txt():
    #----------  Read  nav.dat file (nav aids) ----------
    wptdata         = dict()
    wptdata['wpid']    = []              # identifier (string)
    wptdata['wplat']   = []              # latitude [deg]
    wptdata['wplon']   = []              # longitude [deg]
    wptdata['wptype']  = []              # Type "VOR","NDB","DME","TACAN" or "FIX"

    wptdata['wpelev']  = []              # elevation [m]
    wptdata['wpvar']   = []              # magnetic variation [deg]
    wptdata['wpfreq']  = []              # Navaid frequency kHz(NDB) or MHz(VOR)
    wptdata['wpdesc']  = []              # description


    with open(bs.resource(settings.navdata_path) / 'nav.dat', 'rb') as f:
        print("Reading nav.dat")

        for line in f:
            line = line.decode(encoding="ascii", errors="ignore").strip()
            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example lines:
            # 2  58.61466599  125.42666626    451   522  30    0.0 A    Aldan NDB
            # 3  31.26894444 -085.72630556    334 11120  40   -3.0 OZR  CAIRNS VOR-DME
            # type    lat       lon           elev freq  ?     var id    desc
            #   0      1         2              3    4   5      6   7     8

            fields = line.split()

            # Valid line starst with integers
            if not fields[0].isdigit():
                continue # Next line

            # Get code for type of navaid
            itype = int(fields[0])

            # Type names
            wptypedict = {2:"NDB",3:"VOR",         \
                          4:"ILS",5:"LOC",6:"GS",  \
                          7:"OM",8:"MM",9:"IM",
                          12:"DME",13:"TACAN"}

           # Type code never larger than 20
            if itype not in list(wptypedict.keys()):
                continue # Next line

            wptype = wptypedict[itype]

            # Select types to read
            if wptype not in ["NDB","VOR","DME","TACAN"]:
                continue # Next line

            wptdata["wptype"].append(wptype)

            wptdata["wplat"].append(float(fields[1]))      # latitude [deg]
            wptdata["wplon"].append(float(fields[2]))      # longitude [deg]
            wptdata["wpelev"].append(float(fields[3])*ft)  # elevation [ft]

            if wptype=="NDB":
                wptdata["wpfreq"].append(int(fields[4]))   # NDB freq in kHz

            elif wptype in ["VOR","DME","TACAN"]:
                wptdata["wpfreq"].append(float(fields[4])/100.) # VOR freq in MHz
            else:
                wptdata["wpfreq"].append(0.0)

            if wptype in ["VOR","NDB"]:
                wptdata["wpvar"].append(float(fields[6])) # Magnetic variation in np.degrees
                wptdata["wpid"].append(fields[7]) # Id

            elif wptype in ["DME","TACAN"]:
                wptdata["wpvar"].append(0.0) # Magnetic variation not given
                wptdata["wpid"].append(fields[7]) # Id

            else:
                wptdata['wpvar'].append(0.0) # Magnetic variation in np.degrees
                wptdata['wpid'].append(" ")  # Id

            # Find description
            wpid = wptdata["wpid"][-1]
            try:
                idesc = line.index(wpid)+len(wpid)

                # Description at end of line may include spaces
                wptdata['wpdesc'].append(line[idesc:])  # Description
            except:
                wptdata['wpdesc'].append("   ")  # Description

    #----------  Read  fix.dat file ----------
    with open(bs.resource(settings.navdata_path) / 'fix.dat', 'rb') as f:
        print("Reading fix.dat")
        for line in f:
            line = line.decode(encoding="ascii", errors="ignore").strip()

            # Skip empty lines or comments
            if len(line) < 3 or line[0] == "#":
                continue

            # Start with valid 2 digit latitude -45. or 52.
            if not ((line[0]=="-" and line[3]==".") or line[2]==".")  :
                continue


            # Data line => Process fields of this record, separated by a comma
            # Example line:
            #  30.580372 -094.384169 FAREL
            fields = line.split()

            wptdata["wptype"].append("FIX")


            wptdata['wplat'].append(float(fields[0]))      # latitude [deg]
            wptdata['wplon'].append(float(fields[1]))      # longitude [deg]
            wptdata['wpid'].append(fields[2]) # Id

            # Not given for fixes but fill out tables for equal length
            wptdata['wpelev'].append(0.0)  # elevation [ft]
            wptdata['wpfreq'].append(0.0)  # Fix is no navaid, so no freq
            wptdata['wpvar'].append(0.0)   # Magnetic variation not given
            wptdata['wpdesc'].append("")  # Description

    # Convert lists for lat,lon to numpy-array for vectorised clipping
    wptdata['wplat']   = np.array(wptdata['wplat'])
    wptdata['wplon']   = np.array(wptdata['wplon'])

    #----------  Read  awy.dat file (airway legs) ----------
    awydata   = dict()

    awydata['awid']        = []              # airway identifier (string)
    awydata['awfromwpid']  = []              # from waypoint identifier (string)
    awydata['awfromlat']   = []              # from waypoint lat [deg](float)
    awydata['awfromlon']   = []              # from waypoint lon [deg](float)
    awydata['awtowpid']    = []              # to waypoint identifier (string)
    awydata['awtolat']     = []              # to waypoint lat [deg](float)
    awydata['awtolon']     = []              # to waypoint lon [deg](float)
    awydata['awndir']      = []              # number of directions (1 or 2)
    awydata['awlowfl']     = []              # lowest flight level (int)
    awydata['awupfl']      = []              # highest flight level (int)


    with open(bs.resource(settings.navdata_path) / 'awy.dat', 'rb') as f:
        print("Reading awy.dat")

        for line in f:
            line = line.decode(encoding="ascii", errors="ignore").strip()
            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            fields = line.split()
            if len(fields) < 10:
                continue

            # Example line
            # ABAGO  56.291668  144.236667 GINOL  54.413334  142.011667 1 177 528 A218
            # fromfwp fromlat    fromlon    towp   tolat       tolon   ndir lowfl hghfl airwayid
            #   0        1          2         3      4           5     6     7     8       9

            # Second field should be float
            try:
                fromlat = float(fields[1])
            except:
                continue

            awydata['awfromwpid'].append(fields[0])         # from waypoint identifier (string)
            awydata['awfromlat'].append(fromlat)            # from latitude [deg]
            awydata['awfromlon'].append(float(fields[2]))   # from longitude [deg]

            awydata['awtowpid'].append(fields[3])           # to waypoint identifier (string)
            awydata['awtolat'].append(float(fields[4]))     # to latitude [deg]
            awydata['awtolon'].append(float(fields[5]))     # to longitude [deg]

            awydata['awndir'].append(int(fields[6]))        # number of directions (1 or 2)

            awydata['awlowfl'].append(int(fields[7]))       # number of directions (1 or 2)
            awydata['awupfl'].append(int(fields[8]))        # number of directions (1 or 2)

            if fields[9].find("-")<0:
                #only one airway uses this leg
                awydata['awid'].append(fields[9])
            else:
                # More airways use this leg => copy leg with all airway ids
                awids = fields[9].split("-")
                for i, awid in enumerate(awids):
                    awydata['awid'].append(awid.strip())
                    if i>0:
                        # Repeat last entry
                        for key in awydata:
                            if key!="awid":
                                awydata[key].append(awydata[key][-1])


        # Convert lat,lons to numpy arrays for easy clipping
        awydata['awfromlat'] = np.array(awydata['awfromlat'])
        awydata['awfromlon'] = np.array(awydata['awfromlon'])
        awydata['awtolat']   = np.array(awydata['awtolat'])
        awydata['awtolon']   = np.array(awydata['awtolon'])

    #----------  Read airports.dat file ----------
    aptdata           = dict()
    aptdata['apid']      = []              # 4 char identifier (string)
    aptdata['apname']    = []              # full name
    aptdata['aplat']     = []              # latitude [deg]
    aptdata['aplon']     = []              # longitude [deg]
    aptdata['apmaxrwy']  = []              # reference airport {string}
    aptdata['aptype']    = []              # type (int, 1=large, 2=medium, 3=small)
    aptdata['apco']      = []              # two char country code (string)
    aptdata['apelev']    = []              # field elevation ft-> m
    with open(bs.resource(settings.navdata_path) / 'airports.dat', 'rb') as f:
        types = {'L': 1, 'M': 2, 'S': 3}
        for line in f:
            line = line.decode(encoding="ascii", errors="ignore").strip()
            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example line:
            # EHAM, SCHIPHOL, 52.309, 4.764, Large, 12467, NL
            #  [id]   [name] [lat]    [lon]  [type] [max rwy length in ft] [country code] [elevation]
            #   0        1     2        3       4          5                   6            7
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

            # Not all airports have elevation in data
            try:
                aptdata['apelev'].append(float(fields[7])*ft)  # apt elev [m]
            except:
                aptdata['apelev'].append(0.0)


    aptdata['aplat']    = np.array(aptdata['aplat'])
    aptdata['aplon']    = np.array(aptdata['aplon'])
    aptdata['apmaxrwy'] = np.array(aptdata['apmaxrwy'])
    aptdata['aptype']   = np.array(aptdata['aptype'])
    aptdata['apelev']   = np.array(aptdata['apelev'])

    #----------  Read FIR files ----------
    firdata         = dict()
    firdata['fir']     = []
    firdata['firlat0'] = []
    firdata['firlon0'] = []
    firdata['firlat1'] = []
    firdata['firlon1'] = []

    # Get fir names
    for filname in (bs.resource(settings.navdata_path) / 'fir').iterdir():
        if filname.suffix == '.txt':
            firname = filname.stem
            firdata['fir'].append([firname, [], []])
            with open(filname, 'rb') as f:
                for line in f:
                    rec = line.decode(encoding="ascii", errors="ignore").upper().strip()

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

    #----------  Read ICAO country codes file icao-countries.dat ----------
    codata           = dict()
    codata['coname']   = []              # Country name
    codata['cocode2']  = []              # 2 char code
    codata['cocode3']  = []              # 3 char code
    codata['conr']     = []              # country nr
    with open(bs.resource(settings.navdata_path) / 'icao-countries.dat', 'rb') as f:
        for line in f:
            line = line.decode(encoding="ascii", errors="ignore").strip()
            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Data line: comma separated values:
            # full name, A2 code, A3 code, number

            fields = line.split(",")

            # Skip airports without identifier in file and closed airports
            if fields[0].strip() == "":
                continue

            codata['coname'].append(fields[0].strip())  # id, no leading or trailing spaces
            codata['cocode2'].append(fields[1].strip().upper())  # name, no leading or trailing spaces

            codata['cocode3'].append(fields[2].strip().upper())  # latitude [deg]
            try:
                codata['conr'].append(int(fields[3]))  # longitude [deg]
            except:
                codata['conr'].append(-1)


    return wptdata, aptdata, awydata, firdata, codata


def loadthresholds_txt():
    ''' Runway threshold loader for navdatabase. '''
    rwythresholds = dict()
    curthresholds = None
    zfile = ZipFile(bs.resource(settings.navdata_path) / 'apt.zip')
    print("Reading apt.dat from apt.zip")
    with zfile.open('apt.dat', 'r') as f:
        for line in f:
            elems = line.decode(
                encoding="ascii", errors="ignore").strip().split()
            if len(elems) == 0:
                continue

            # 1: AIRPORT
            if elems[0] == '1':
                # Add airport to runway threshold database
                curthresholds = dict()
                rwythresholds[elems[4]] = curthresholds
                continue

            if elems[0] == '100':
                # Only asphalt and concrete runways
                if int(elems[2]) > 2:
                    continue
                # rwy_lbl = (elems[8], elems[17])

                lat0 = float(elems[9])
                lon0 = float(elems[10])
                offset0 = float(elems[11])

                lat1 = float(elems[18])
                lon1 = float(elems[19])
                offset1 = float(elems[20])

                # threshold information: ICAO code airport, Runway identifier,
                # latitude, longitude, bearing
                # vertices: gives vertices of the box around the threshold

                # opposite runways are on the same line. RWY1: 8-11, RWY2: 17-20
                # Hence, there are two thresholds per line
                # thr0: First lat0 and lon0 , then lat1 and lat1, offset=[11]
                # thr1: First lat1 and lat1 , then lat0 and lon0, offset=[20]

                thr0 = thresholds(np.radians(lat0), np.radians(lon0),
                                  np.radians(lat1), np.radians(lon1), offset0)
                thr1 = thresholds(np.radians(lat1), np.radians(lon1),
                                  np.radians(lat0), np.radians(lon0), offset1)
                curthresholds[elems[8]] = thr0
                curthresholds[elems[17]] = thr1
                continue
    return rwythresholds


def thresholds(lat1, lon1, lat2, lon2, offset):
    ''' calculates the threshold points per runway
        underlying equations can be found at
        http://www.movable-type.co.uk/scripts/latlong.html '''

    d = offset * REARTH_INV
    deltal = lon2 - lon1

    # calculate runway bearing
    bearing = np.arctan2(np.sin(deltal) * np.cos(lat2), (np.cos(lat1) * np.sin(lat2) -
                                              np.sin(lat1) * np.cos(lat2) * np.cos(deltal)))

    # normalize to 0-360 degrees
    bearing = np.radians((np.degrees(bearing) + 360) % 360)

    # get threshold points
    latthres, lonthres = thrpoints(lat1, lon1, d, bearing)

    return np.degrees(latthres), np.degrees(lonthres), np.degrees(bearing)


def thrpoints(lat1, lon1, d, bearing):
    ''' Calculate threshold points as well as end points of threshold box
    underlying equations can be found at
    http://www.movable-type.co.uk/scripts/latlong.html '''
    latthres = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(bearing))

    lonthres = lon1 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(lat1),
                            np.cos(d) - np.sin(lat1) * np.sin(latthres))

    return latthres, lonthres
