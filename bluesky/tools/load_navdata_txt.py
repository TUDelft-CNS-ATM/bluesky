from ..settings import data_path


def load_navdata_txt():
    #---------- Read waypoints.dat file ----------
    # Create empty database
    wpid    = []              # identifier (string)
    wplat   = []              # latitude [deg]
    wplon   = []              # longitude [deg]
    wpapt   = []              # reference airport {string}
    wptype  = []              # type (string)
    wpco    = []              # two char country code (string)
    wpswbmp = []              # switch indicating whether label bmp is present
    wplabel = []              # List to store bitmaps of label
    with open(data_path + "/global/waypoints.dat","r") as f:
        for line in f:

    

    # Process lines to fill database
    for line in lines:

        # Skip empty lines or comments
        if line.strip()=="":
            continue
        elif line.strip()[0]=="#":
            continue
        
        # Data line => Process fields of this record, separated by a comma
        # Example line:
        # ABARA, , 61.1833, 50.85, UUYY, High and Low Level, RS
        #  [id]    [lat]    [lon]  [airport]  [type] [country code]
        #   0  1     2       3         4        5         6
   
        fields = line.split(",")
        wpid.append(fields[0].strip())  # id, no leading or trailing spaces

        wplat.append(float(fields[2]))  # latitude [deg]
        wplon.append(float(fields[3]))  # longitude [deg]

        wpapt.append(fields[4].strip())  # id, no leading or trailing spaces
        wptype.append(fields[5].strip().lower())    # type
        wpco.append(fields[6].strip())     # country code
        wpswbmp.append(False)     # country code

    wplat = np.array(wplat)
    wplon = np.array(wplon)
    wplabel   = len(wpid)*[0]         # list to store bitmaps           

    print "    ",len(wpid),"waypoints read."

    #----------  Read airports.dat file ----------
    print "Reading airports.dat from",subfolder

    # Read data into list of ascii lines
    path = "./data/"+subfolder+"/"
    # f = open(path+"airports.dat","r")
    f = open(airport_file, 'r')
    lines = f.readlines()
    f.close()

    # Create empty database
    apid      = []              # 4 char identifier (string)
    apname    = []              # full name
    aplat     = []              # latitude [deg]
    aplon     = []              # longitude [deg]
    apmaxrwy  = []              # reference airport {string}
    aptype    = []              # type (int, 1=large, 2=medium, 3=small)
    apco      = []              # two char country code (string)
    apswbmp   = []              # switch indicating whether label bmp is present
    aplabel   = []              # list to store bitmaps           

    # Process lines to fill database
    types = {'L': 1, 'M': 2, 'S': 3}
    for line in lines:
        # Skip empty lines or comments
        if line.strip()=="":
            continue
        elif line.strip()[0]=="#":
            continue
        
        # Data line => Process fields of this record, separated by a comma
        # Example line:
        # EHAM, SCHIPHOL, 52.309, 4.764, Large, 12467, NL
        #  [id]   [name] [lat]    [lon]  [type] [max rwy length in ft] [country code]
        #   0        1     2        3       4          5                   6
   
        fields = line.split(",")

        # Skip airports without identifier in file and closed airports
        if fields[0].strip()=="" or fields[4].strip() == 'Closed':
            continue

        # print fields[0]
        
        apid.append(fields[0].strip())  # id, no leading or trailing spaces
        apname.append(fields[1].strip())  # name, no leading or trailing spaces

        aplat.append(float(fields[2]))  # latitude [deg]
        aplon.append(float(fields[3]))  # longitude [deg]

        aptype.append(types[fields[4].strip()[0]])  # large=1, medium=2, small=3

        # Not all airports have rwy length (e.g. heliports)
        try:
            apmaxrwy.append(float(fields[5])*ft)  # max rwy ltgh [m]
        except:
            apmaxrwy.append(0.0)
        
      
        apco.append(fields[6].strip().lower()[:2])     # country code
        apswbmp.append(False)     # country code

    aplabel   = len(apid)*[0]         # list to store bitmaps           

    aplat    = np.array(aplat)
    aplon    = np.array(aplon)
    apmaxrwy = np.array(apmaxrwy)
    aptype   = np.array(aptype)

    print "    ",len(apid),"airports read."

    fir = []
    firlat0 = []
    firlon0 = []
    firlat1 = []
    firlon1 = []
    
    # Check whether fir subfolder exists
    # try:
    if True:            
        files = os.listdir(path+"fir")
        print "Reading fir subfolder",

        # Get fir names
        for filname in files:
            if filname.count(".txt")>0:

                firname = filname[:filname.index(".txt")]

                fir.append([firname,[],[]])

                f = open(path+"fir/"+filname,"r")
                lines = f.readlines()

                # Read lines: >N049.28.00.000 E006.20.00.000
                for line in lines:
                    rec = line.upper().strip()
                    if rec=="":
                        continue
                    latsign = 2*int(line[0]=="N")-1
                    latdeg  = float(line[1:4])
                    latmin  = float(line[5:7])
                    latsec  = float(line[8:14])
                    lat     = latsign*latdeg+latmin/60.+latsec/3600.

                    lonsign = 2*int(line[15]=="E")-1
                    londeg  = float(line[16:19])
                    lonmin  = float(line[20:22])
                    lonsec  = float(line[23:29])
                    lon     = lonsign*londeg+lonmin/60.+lonsec/3600.

                    # For drawing create a line from last lat,lon to current lat,lon
                    if len(fir[-1][1])>0:  # skip first lat,lon
                       firlat0.append(fir[-1][1][-1])                            
                       firlon0.append(fir[-1][2][-1])
                       firlat1.append(lat)                            
                       firlon1.append(lon)
                    
                    # Add to FIR record
                    fir[-1][1].append(lat)                        
                    fir[-1][2].append(lon)

        # Convert lat/lon lines to numpy arrays 
        firlat0 = np.array(firlat0)
        firlat1 = np.array(firlat1)
        firlon0 = np.array(firlon0)
        firlon1 = np.array(firlon1)

        print len(fir)," FIRS read."
        # No fir folders found or error in reading fir files:
    else:
    # except:
        print "No fir folder in",path