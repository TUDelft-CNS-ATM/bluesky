''' BlueSky synthetic conflict generator plugin. '''
import random
import numpy as np
from bluesky import stack, traf, sim
from bluesky.tools.aero import ft, eas2tas


def init_plugin():
    ''' Plugin initialisation function. '''

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SYNTHETIC',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config


@stack.commandgroup(aliases=('SYNTHETIC',))
def syn():
    ''' SYNTHETIC: Generate synthetic conflict geometries. '''
    return True, ("This is the synthetic traffic scenario module\n"
                  "Possible subcommands: SIMPLE, SIMPLED, DIFG, SUPER, SPHERE, "
                  "MATRIX, FLOOR, TAKEOVER, WALL, ROW, COLUMN, DISP")


@syn.subcommand
def simple():
    ''' SIMPLE: Generate a simple 2-aircraft conflict. '''
    sim.reset()
    traf.cre(acid="OWNSHIP", actype="GENERIC", aclat=-.5, aclon=0,
             achdg=0, acalt=5000 * ft, acspd=200)
    traf.cre(acid="INTRUDER", actype="GENERIC", aclat=0, aclon=.5,
             achdg=270, acalt=5000 * ft, acspd=200)
    return True


@syn.subcommand
def simpled():
    ''' SIMPLED: Generate a simple 2-aircraft conflict
        with random speed and distance. '''
    sim.reset()
    ds = random.uniform(0.92, 1.08)
    dd = random.uniform(0.92, 1.08)
    traf.cre(acid="OWNSHIP", actype="GENERIC", aclat=-.5 * dd, aclon=0,
             achdg=0, acalt=20000 * ft, acspd=200 * ds)
    traf.cre(acid="INTRUDER", actype="GENERIC", aclat=0, aclon=.5 / dd,
             achdg=270, acalt=20000 * ft, acspd=200 / ds)
    return True


@syn.subcommand(name='SUPER')
def gensuper(numac:int):
    ''' SUPER: Generate a circular super conflict.

        Arguments:
        -numac: The number of aircraft in the conflict.
    '''
    sim.reset()

    distance=0.50 #this is in degrees lat/lon, for now
    alt=20000*ft #ft
    spd=200 #kts
    for i in range(numac):
        angle=2*np.pi/numac*i
        acid = "SUP" + str(i)
        traf.cre(acid=acid, actype="SUPER",
                 aclat=distance * -np.cos(angle),
                 aclon=distance * np.sin(angle),
                 achdg=360.0 - 360.0 / numac * i,
                 acalt=alt, acspd=spd)
    return True


@syn.subcommand
def sphere(numac:int):
    ''' SUPER: Generate a spherical super conflict,
        with 3 layers of superconflicts.
        Arguments:
        -numac: The number of aircraft in each layer.
    '''
    sim.reset()

    distance = 0.5  # this is in degrees lat/lon, for now
    distancenm = distance*111319./1852
    alt = 20000  # ft
    spd = 150  # kts
    vs = 4  # m/s
    timetoimpact = distancenm/spd*3600  # seconds
    altdifference = vs*timetoimpact  # m
    midalt = alt
    lowalt = alt-altdifference
    highalt = alt+altdifference
    hispd = eas2tas(spd, highalt)
    mispd = eas2tas(spd, midalt)
    lospd = eas2tas(spd, lowalt)
    hispd = spd
    mispd = spd
    lospd = spd
    for i in range(numac):
        angle = np.pi*(2./numac*i)
        lat = distance*-np.cos(angle)
        lon = distance*np.sin(angle)
        track = np.degrees(-angle)

        acidl = "SPH"+str(i)+"LOW"
        traf.cre(acid=acidl, actype="SUPER", aclat=lat, aclon=lon,
                 achdg=track, acalt=lowalt*ft, acspd=lospd)
        acidm = "SPH"+str(i)+"MID"
        traf.cre(acid=acidm, actype="SUPER", aclat=lat, aclon=lon,
                 achdg=track, acalt=midalt*ft, acspd=mispd)
        acidh = "SPH"+str(i)+"HIG"
        traf.cre(acid=acidh, actype="SUPER", aclat=lat, aclon=lon,
                 achdg=track, acalt=highalt*ft, acspd=hispd)

        idxl = traf.id.index(acidl)
        idxh = traf.id.index(acidh)

        traf.vs[idxl] = vs
        traf.vs[idxh] = -vs

        traf.selvs[idxl] = vs
        traf.selvs[idxh] = -vs

        traf.selalt[idxl] = highalt
        traf.selalt[idxh] = lowalt
    return True


@syn.subcommand
def funnel(size:int):
    ''' FUNNEL: create a funnel conflict. '''
    sim.reset()
    # traf.asas=CASASfunnel.Dbconf(300., 5.*nm, 1000.*ft)
    mperdeg=111319.
    distance=0.90 #this is in degrees lat/lon, for now
    alt=20000 #meters
    spd=200 #kts
    numac=8 #number of aircraft
    for i in range(numac):
        angle=np.pi/2/numac*i+np.pi/4
        acid="SUP"+str(i)
        traf.cre(acid=acid, actype="SUPER",
                 aclat=distance*-np.cos(angle),
                 aclon=distance*-np.sin(angle),
                 achdg=90, acalt=alt, acspd=spd)

    # the factor 1.01 is so that the funnel doesn't collide with itself
    separation=traf.cd.rpz*1.01 #[m]
    sepdeg=separation/np.sqrt(2.)/mperdeg #[deg]

    for f_row in range(1):
        for f_col in range(15):
            opening=(size+1)/2.*separation/mperdeg
            Coldeg=sepdeg*f_col  #[deg]
            Rowdeg=sepdeg*f_row  #[deg]
            acid1="FUNN"+str(f_row)+"-"+str(f_col)
            acid2="FUNL"+str(f_row)+"-"+str(f_col)
            traf.cre(acid=acid1, actype="FUNNEL",
                     aclat=Coldeg+Rowdeg+opening,
                     aclon=-Coldeg+Rowdeg+0.5,
                     achdg=0, acalt=alt, acspd=0)
            traf.cre(acid=acid2, actype="FUNNEL",
                     aclat=-Coldeg-Rowdeg-opening,
                     aclon=-Coldeg+Rowdeg+0.5,
                     achdg=0, acalt=alt, acspd=0)

@syn.subcommand
def matrix(size:int):
    ''' MATRIX: create a conflict with several aircraft
        flying in a matrix formation.
    '''
    sim.reset()
    mperdeg = 111319.
    hsep = traf.cd.rpz  # [m] horizontal separation minimum
    hseplat = hsep/mperdeg
    matsep = 1.1  # factor of extra space in the matrix
    hseplat = hseplat*matsep
    vel = 200  # m/s
    # degrees latlon flown in 5 minutes
    extradist = (vel*1.1)*5*60/mperdeg
    for i in range(size):
        acidn = "NORTH"+str(i)
        traf.cre(acid=acidn, actype="MATRIX",
                 aclat=hseplat*(size-1.)/2+extradist,
                 aclon=(i-(size-1.)/2)*hseplat,
                 achdg=180, acalt=20000*ft, acspd=vel)
        acids = "SOUTH"+str(i)
        traf.cre(acid=acids, actype="MATRIX",
                 aclat=-hseplat*(size-1.)/2-extradist,
                 aclon=(i-(size-1.)/2)*hseplat,
                 achdg=0, acalt=20000*ft, acspd=vel)
        acide = "EAST"+str(i)
        traf.cre(acid=acide, actype="MATRIX",
                 aclat=(i-(size-1.)/2)*hseplat,
                 aclon=hseplat*(size-1.)/2+extradist,
                 achdg=270, acalt=20000*ft, acspd=vel)
        acidw = "WEST"+str(i)
        traf.cre(acid=acidw, actype="MATRIX",
                 aclat=(i-(size-1.)/2)*hseplat,
                 aclon=-hseplat*(size-1.)/2-extradist,
                 achdg=90, acalt=20000*ft, acspd=vel)


@syn.subcommand
def floor():
    ''' FLOOR: create a conflict with several aircraft flying
        in a floor formation.
    '''
    sim.reset()
    mperdeg = 111319.
    altdif = 3000  # ft
    hsep = traf.cd.rpz  # [m] horizontal separation minimum
    floorsep = 1.1  # factor of extra spacing in the floor
    hseplat = hsep/mperdeg*floorsep
    traf.cre(acid="OWNSHIP", actype="FLOOR",
             aclat=-1, aclon=0,
             achdg=90, acalt=(20000+altdif)*ft, acspd=200)
    idx = traf.id.index("OWNSHIP")
    traf.selvs[idx] = -10
    traf.selalt[idx] = 20000-altdif
    for i in range(20):
        acid = "OTH"+str(i)
        traf.cre(acid=acid, actype="FLOOR",
                 aclat=-1, aclon=(i-10)*hseplat,
                 achdg=90, acalt=20000*ft, acspd=200)
    return True


@syn.subcommand
def takeover(numac:int):
    ''' TAKEOVER: create a conflict with several aircraft overtaking eachother
    '''
    sim.reset()
    mperdeg = 111319.
    vsteps = 50  # [m/s]
    for v in range(vsteps, vsteps*(numac+1), vsteps):  # m/s
        acid = "OT"+str(v)
        distancetofly = v*5*60  # m
        degtofly = distancetofly/mperdeg
        traf.cre(acid=acid, actype="OT", aclat=0, aclon=-degtofly,
                 achdg=90, acalt=20000*ft, acspd=v)
    return True


@syn.subcommand
def wall():
    ''' WALL: create a conflict with several aircraft flying in a wall formation
    '''
    sim.reset()

    mperdeg = 111319.
    distance = 0.6  # in degrees lat/lon, for now
    hsep = traf.cd.rpz  # [m] horizontal separation minimum
    hseplat = hsep/mperdeg
    wallsep = 1.1  # factor of extra space in the wall
    traf.cre(acid="OWNSHIP", actype="WALL",
             aclat=0, aclon=-distance,
             achdg=90, acalt=20000*ft, acspd=200)
    for i in range(20):
        acid = "OTHER"+str(i)
        traf.cre(acid=acid, actype="WALL",
                 aclat=(i-10)*hseplat*wallsep, aclon=distance,
                 achdg=270, acalt=20000*ft, acspd=200)

        return True


@syn.subcommand
def row(numac:int, angle:int, radius:float=1.0, alt:'alt'=1e5,
        spd:'spd'=300.0, actype:'txt'='B747'):
    ''' ROW: create a conflict with several aircraft flying in two rows
        angled towards each other.
    '''
    sim.reset()

    mperdeg = 111319.
    hsep = traf.cd.rpz  # [m] horizontal separation minimum
    hseplat = hsep/mperdeg
    matsep = 1.1  # factor of extra space in the formation
    hseplat = hseplat*matsep

    aclat = radius * np.cos(np.deg2rad(angle))  # [deg]
    aclon = radius * np.sin(np.deg2rad(angle))
    latsep = abs(hseplat*np.cos(np.deg2rad(90-angle)))  # [deg]
    lonsep = abs(hseplat*np.sin(np.deg2rad(90-angle)))

    alternate = 1
    for i in range(numac):  # Create a/c
        aclat = aclat+i*latsep*alternate
        aclon = aclon-i*lonsep*alternate
        traf.cre(acid="ANG"+str(i*2), actype=actype,
                 aclat=aclat, aclon=aclon,
                 achdg=180+angle, acalt=alt*ft, acspd=spd)
        traf.cre(acid="ANG"+str(i*2+1), actype=actype,
                 aclat=aclat, aclon=-aclon,
                 achdg=180-angle, acalt=alt*ft, acspd=spd)
        alternate = alternate * -1

    return True


@syn.subcommand
def col(numac:int, angle:int, radius:float=1.0, alt:'alt'=1e5,
        spd:'spd'=300.0, actype:'txt'='B747'):
    ''' COL: create a conflict with several aircraft flying in two columns
        angled towards each other.
    '''
    sim.reset()

    mperdeg = 111319.
    hsep = traf.cd.rpz  # [m] horizontal separation minimum
    hseplat = hsep/mperdeg
    matsep = 1.1  # factor of extra space in the formation
    hseplat = hseplat*matsep

    aclat = radius * np.cos(np.deg2rad(angle))  # [deg]
    aclon = radius * np.sin(np.deg2rad(angle))
    latsep = abs(hseplat*np.cos(np.deg2rad(angle)))  # [deg]
    lonsep = abs(hseplat*np.sin(np.deg2rad(angle)))

    traf.cre(acid="ANG0", actype=actype,
             aclat=aclat, aclon=aclon,
             achdg=180+angle, acalt=alt*ft, acspd=spd)
    traf.cre(acid="ANG1", actype=actype,
             aclat=aclat, aclon=-aclon,
             achdg=180-angle, acalt=alt*ft, acspd=spd)

    for i in range(1, numac):  # Create a/c
        aclat = aclat+latsep
        aclon = aclon+lonsep
        traf.cre(acid="ANG"+str(i*2), actype=actype,
                 aclat=aclat, aclon=aclon,
                 achdg=180+angle, acalt=alt*ft, acspd=spd)
        traf.cre(acid="ANG"+str(i*2+1), actype=actype,
                 aclat=aclat, aclon=-aclon,
                 achdg=180-angle, acalt=alt*ft, acspd=spd)
